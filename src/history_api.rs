use std::{
    collections::{HashMap, HashSet},
    fs,
    io::BufRead,
    path::Path,
    str::FromStr,
};

use ark_bls12_381::{Fr, G1Projective as G1};
use eyre::{bail, Result};
use ethers_core::types::Address;

use crate::{
    codec::{fr_from_hex, fr_to_hex, g1_from_hex, g1_to_hex},
    history::vupdate_history_vupdate_dc,
    proof_history_core::{
        canonicalize_beta_delta, for_each_journal_line_rev_until, pinned_commitment_at,
    },
    types::{HistoryOp, JournalLineLite, ProofServerHistoryOut},
    vc_context::VcContext,
    Indexer, ProofServerState, SnapshotOut, StateTracker,
};

fn read_addresses_file(path: &Path) -> eyre::Result<Vec<Address>> {
    let f = fs::File::open(path)?;
    let rd = std::io::BufReader::new(f);
    let mut out = Vec::new();

    for (i, line) in rd.lines().enumerate() {
        let s = line?;
        let s = s.trim();
        if s.is_empty() {
            continue;
        }
        let a = Address::from_str(s)
            .map_err(|e| eyre::eyre!("bad addr line {}: {} ({})", i + 1, s, e))?;
        out.push(a);
    }
    Ok(out)
}

pub fn values_at_block_for_indices_hex(
    dataset_dir: &Path,
    snapshot_path: &Path,
    snapshot_vals_path: &Path,
    universe_file: &Path,
    target_block: u64,
    indices: &[usize],
) -> Result<Vec<String>> {
    let snap: SnapshotOut = serde_json::from_slice(&fs::read(snapshot_path)?)?;

    if target_block < snap.block_number {
        bail!(
            "target_block {} < snapshot block {}",
            target_block,
            snap.block_number
        );
    }

    let universe = read_addresses_file(universe_file)?;
    let _indexer = Indexer::from_universe_sequential(&universe, snap.n)?;

    for &idx in indices {
        if idx >= universe.len() {
            bail!("index {} out of universe bounds {}", idx, universe.len());
        }
    }

    let mut tracker = StateTracker::from_snapshot_vals_file(
        dataset_dir,
        100_000,
        snap.block_number,
        &universe,
        snapshot_vals_path,
    )?;

    while tracker.cur_block < target_block {
        tracker.apply_next_block()?;
    }

    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        let addr = universe[idx];
        let val: &Fr = tracker
            .balances
            .get(&addr)
            .ok_or_else(|| eyre::eyre!("missing balance for address {:?}", addr))?;
        out.push(fr_to_hex(val));
    }

    Ok(out)
}

pub fn proof_server_history_loaded_json(
    ctx: &VcContext,
    ps: &ProofServerState,
    journal: &Path,
    dataset_dir: &Path,
    snapshot_path: &Path,
    snapshot_vals_path: &Path,
    universe_file: &Path,
    start_block: u64,
    mut blocks: Vec<u64>,
) -> Result<Vec<u8>> {
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

    if ps.alpha_indices.len() != ps.alpha_witnesses_hex.len() {
        bail!("alpha_indices / alpha_witnesses_hex length mismatch");
    }

    let alpha = ps.alpha_indices.clone();

    let mut gq_head: Vec<G1> = Vec::with_capacity(ps.alpha_witnesses_hex.len());
    for hx in &ps.alpha_witnesses_hex {
        gq_head.push(g1_from_hex(hx)?);
    }

    let head_block = ps.last_block;
    let min_block = blocks[0];
    let want: HashSet<u64> = blocks.iter().copied().collect();

    let mut stream_rev: Vec<HistoryOp> = Vec::new();
    let mut expected = head_block;

    for_each_journal_line_rev_until(journal, |line| -> Result<bool> {
        if expected <= min_block {
            return Ok(false);
        }

        let j: JournalLineLite = serde_json::from_str(line)?;

        if j.block_number <= start_block || j.block_number > head_block {
            return Ok(true);
        }
        if j.srs_id != ps.srs_id || j.logn != ps.logn || j.n != ps.n {
            return Ok(true);
        }

        if j.block_number < expected {
            bail!(
                "journal gap: missing block {} while building history stream (next seen block {})",
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

        let mut delta_neg = Vec::<Fr>::with_capacity(j.delta_hex.len());
        for hx in &j.delta_hex {
            delta_neg.push(-fr_from_hex(hx)?);
        }

        let (beta, delta) = canonicalize_beta_delta(&j.changed_indices, &delta_neg);

        if !beta.is_empty() {
            stream_rev.push(HistoryOp::Update { beta, delta });
        }

        expected -= 1;
        Ok(true)
    })?;

    if expected > min_block {
        bail!(
            "journal ended early while building history stream: stopped at {}, need min_block {}",
            expected,
            min_block
        );
    }

    if want.contains(&min_block) {
        stream_rev.push(HistoryOp::Query {
            query_id: min_block as usize,
        });
    }

    let vu = vupdate_history_vupdate_dc(ctx, &alpha, &gq_head, &stream_rev);

    let mut by_block: HashMap<u64, Vec<G1>> = HashMap::new();
    for r in vu {
        by_block.insert(r.query_id as u64, r.proofs);
    }

    let mut out_all = Vec::<ProofServerHistoryOut>::new();
    for &b in &blocks {
        let wits = by_block
            .get(&b)
            .ok_or_else(|| eyre::eyre!("missing block {}", b))?;

        let gc_hex = pinned_commitment_at(journal, &ps.srs_id, ps.n, ps.logn, b)?;
        let values_hex = values_at_block_for_indices_hex(
            dataset_dir,
            snapshot_path,
            snapshot_vals_path,
            universe_file,
            b,
            &alpha,
        )?;

        out_all.push(ProofServerHistoryOut {
            user_id: "loaded_profile".to_string(),
            block: b,
            indices: alpha.clone(),
            values_hex,
            witnesses_hex: wits.iter().map(g1_to_hex).collect(),
            gc_hex,
        });
    }

    out_all.sort_by_key(|o| o.block);

    Ok(serde_json::to_vec_pretty(&out_all)?)
}