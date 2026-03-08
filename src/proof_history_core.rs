use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    time::Instant,
};

use ark_bls12_381::{Fr, G1Projective as G1};
use ark_ff::{Field, Zero};
use eyre::{bail, Result};

use crate::{
    codec::{fr_from_hex, g1_from_hex, g1_to_hex},
    history::vupdate_history_vupdate_dc,
    journal,
    types::{HistoryOp, JournalLineLite, Metric, ProofServerHistoryOut},
    vc_context::VcContext,
    JournalLine, ProofServerState,
};

fn t_start() -> Instant {
    Instant::now()
}

fn t_us(t0: Instant) -> u128 {
    t0.elapsed().as_micros()
}

/// Emit metrics to stderr
fn emit(
    phase: &'static str,
    block: u64,
    n: usize,
    alpha: usize,
    beta: usize,
    micros: u128,
) {
    let mut err = std::io::stderr();
    emit_to(&mut err, phase, block, n, alpha, beta, micros);
}

/// Emit metrics to an arbitrary writer
fn emit_to<W: Write>(
    w: &mut W,
    phase: &'static str,
    block: u64,
    n: usize,
    alpha: usize,
    beta: usize,
    micros: u128,
) {
    let m = Metric {
        phase,
        block,
        n,
        alpha,
        beta,
        micros,
    };
    let _ = writeln!(w, "METRIC {}", serde_json::to_string(&m).unwrap());
}

pub fn for_each_journal_line_rev_until<F>(path: &Path, mut f: F) -> Result<()>
where
    F: FnMut(&str) -> Result<bool>,
{
    const CHUNK_MIN: usize = 256 * 1024;
    const CHUNK_MAX: usize = 8 * 1024 * 1024;

    let mut file = fs::File::open(path)?;
    let mut pos = file.seek(SeekFrom::End(0))?;

    let mut carry = String::new();
    let mut chunk_size = CHUNK_MIN;
    let mut iters: u64 = 0;

    while pos > 0 {
        iters += 1;

        let rd = std::cmp::min(chunk_size as u64, pos) as usize;
        pos -= rd as u64;
        file.seek(SeekFrom::Start(pos))?;

        let mut buf = vec![0u8; rd];
        file.read_exact(&mut buf)?;

        let mut chunk = String::from_utf8_lossy(&buf).to_string();
        chunk.push_str(&carry);

        let mut parts: Vec<&str> = chunk.split('\n').collect();
        carry = parts.remove(0).to_string();

        for line in parts.into_iter().rev() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if !f(line)? {
                return Ok(());
            }
        }

        if pos > (chunk_size as u64) * 4 && chunk_size < CHUNK_MAX {
            chunk_size = std::cmp::min(chunk_size * 2, CHUNK_MAX);
        } else if iters > 64 && chunk_size < CHUNK_MAX {
            chunk_size = std::cmp::min(chunk_size * 2, CHUNK_MAX);
        }
    }

    let line = carry.trim();
    if !line.is_empty() {
        let _ = f(line)?;
    }

    Ok(())
}

pub fn cancelled(cancel: Option<&Path>) -> bool {
    cancel.map(|p| p.exists()).unwrap_or(false)
}

pub fn canonicalize_beta_delta(beta: &[usize], delta: &[Fr]) -> (Vec<usize>, Vec<Fr>) {
    let mut acc: BTreeMap<usize, Fr> = BTreeMap::new();

    for (&i, &d) in beta.iter().zip(delta.iter()) {
        if !d.is_zero() {
            *acc.entry(i).or_insert(Fr::ZERO) += d;
        }
    }

    acc.retain(|_, v| !v.is_zero());
    acc.into_iter().unzip()
}

pub fn pinned_commitment_at(
    journal_path: &Path,
    srs_id: &str,
    n: usize,
    logn: usize,
    block: u64,
) -> Result<String> {
    let mut out: Option<String> = None;

    journal::for_each_line_filtered(journal_path, srs_id, n, logn, |j: JournalLine| {
        if j.block_number == block {
            if let Some(gc) = j.gc_hex.clone() {
                out = Some(gc);
            } else {
                bail!("journal line for block {} missing pinned gc_hex", block);
            }
        }
        Ok(())
    })?;

    out.ok_or_else(|| eyre::eyre!("no journal entry for block {}", block))
}

pub fn proof_server_history_core(
    ctx: &VcContext,
    ps: &ProofServerState,
    journal_path: &Path,
    start_block: u64,
    mut blocks: Vec<u64>,
    user_id: String,
    out: std::path::PathBuf,
    cancel: Option<&Path>,
) -> eyre::Result<()> {
    let t_total = t_start();
    let n = ps.n;

    if blocks.is_empty() {
        bail!("blocks empty");
    }

    blocks.sort_unstable();
    blocks.dedup();

    if blocks[0] <= start_block {
        bail!("query block <= start_block");
    }
    if *blocks.last().unwrap() > ps.last_block {
        bail!("query beyond head");
    }

    let t_head = t_start();
    let (user_alpha, user_pos_in_union) = ps
        .user_alpha(&user_id)
        .ok_or_else(|| eyre::eyre!("unknown user_id"))?;

    let mut user_gq_head: Vec<G1> = Vec::with_capacity(user_pos_in_union.len());
    for &p in &user_pos_in_union {
        user_gq_head.push(g1_from_hex(&ps.alpha_witnesses_hex[p])?);
    }

    emit(
        "[HistoryHeadUserDecode]",
        ps.last_block,
        n,
        user_alpha.len(),
        user_alpha.len(),
        t_us(t_head),
    );

    let head_block = ps.last_block;
    let min_block = blocks[0];
    let want: HashSet<u64> = blocks.iter().copied().collect();

    let t_build = t_start();
    let mut stream_rev: Vec<HistoryOp> = Vec::new();
    let mut expected = head_block;

    let mut lines_parsed: usize = 0;
    let mut micros_json_parse: u128 = 0;
    let mut micros_delta_decode: u128 = 0;

    for_each_journal_line_rev_until(journal_path, |line| -> eyre::Result<bool> {
        if cancelled(cancel) {
            eyre::bail!("cancelled during journal scan");
        }

        if expected <= min_block {
            return Ok(false);
        }

        let t_j = t_start();
        let j: JournalLineLite = serde_json::from_str(line)?;
        micros_json_parse += t_us(t_j);
        lines_parsed += 1;

        if j.block_number <= start_block || j.block_number > head_block {
            return Ok(true);
        }
        if j.srs_id != ps.srs_id || j.logn != ps.logn || j.n != ps.n {
            return Ok(true);
        }

        if j.block_number < expected {
            eyre::bail!(
                "journal gap: missing block {} while building history stream (next seen block {}). \
                 Journal must contain every block in the window (even if beta empty).",
                expected,
                j.block_number
            );
        }
        if j.block_number > expected {
            return Ok(true);
        }

        let b = expected;

        if want.contains(&b) {
            stream_rev.push(HistoryOp::Query {
                query_id: b as usize,
            });
        }

        let t_d = t_start();
        let mut delta_neg = Vec::<Fr>::with_capacity(j.delta_hex.len());
        for hx in &j.delta_hex {
            delta_neg.push(-fr_from_hex(hx)?);
        }
        micros_delta_decode += t_us(t_d);

        let (beta, delta) = canonicalize_beta_delta(&j.changed_indices, &delta_neg);

        if !beta.is_empty() {
            stream_rev.push(HistoryOp::Update { beta, delta });
        }

        expected -= 1;
        Ok(true)
    })?;

    if expected > min_block {
        eyre::bail!(
            "journal ended early while building history stream: stopped at {}, need min_block {}. \
             Journal is incomplete for the requested window.",
            expected,
            min_block
        );
    }

    if want.contains(&min_block) {
        stream_rev.push(HistoryOp::Query {
            query_id: min_block as usize,
        });
    }

    emit(
        "[HistoryJournalJsonParseLite]",
        0,
        n,
        user_alpha.len(),
        lines_parsed,
        micros_json_parse,
    );
    emit(
        "[HistoryDeltaDecode]",
        0,
        n,
        user_alpha.len(),
        stream_rev.len(),
        micros_delta_decode,
    );
    emit(
        "[HistoryBuildStream]",
        0,
        n,
        user_alpha.len(),
        stream_rev.len(),
        t_us(t_build),
    );

    if cancelled(cancel) {
        eyre::bail!("cancelled before vupdate");
    }

    let t_vu = t_start();
    let vu = vupdate_history_vupdate_dc(ctx, &user_alpha, &user_gq_head, &stream_rev);
    emit(
        "[HistoryVUpdateUserAlpha]",
        0,
        n,
        user_alpha.len(),
        vu.len(),
        t_us(t_vu),
    );

    let t_payload = t_start();
    let mut by_block: HashMap<u64, Vec<G1>> = HashMap::new();
    for r in vu {
        by_block.insert(r.query_id as u64, r.proofs);
    }

    let mut out_all = Vec::<ProofServerHistoryOut>::new();
    for &b in &blocks {
        let wits = by_block
            .get(&b)
            .ok_or_else(|| eyre::eyre!("missing block {}", b))?;

        let gc_hex = pinned_commitment_at(journal_path, &ps.srs_id, ps.n, ps.logn, b)?;

        out_all.push(ProofServerHistoryOut {
            user_id: user_id.clone(),
            block: b,
            indices: user_alpha.clone(),
            values_hex: vec![],
            witnesses_hex: wits.iter().map(g1_to_hex).collect(),
            gc_hex,
        });
    }

    out_all.sort_by_key(|o| o.block);

    let json = serde_json::to_vec_pretty(&out_all)?;
    fs::write(&out, &json)?;
    emit(
        "[HistoryPayloadWrite]",
        0,
        n,
        user_alpha.len(),
        json.len(),
        t_us(t_payload),
    );

    emit("[HistoryTotal]", 0, n, user_alpha.len(), 0, t_us(t_total));
    Ok(())
}