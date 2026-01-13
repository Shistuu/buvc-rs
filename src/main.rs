// src/main.rs
use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::{Field, Zero};
use clap::{Parser, Subcommand};
use ethers_core::types::{Address};
use eyre::{bail, Result};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use buvc_rs::codec::{
    delta_fr, fr_from_hex, fr_from_u256_exact, fr_to_hex, g1_from_hex, g1_to_hex, parse_addr_list,
};
use buvc_rs::dataset_helpers::StateTracker;
use buvc_rs::history::{vupdate_history_same_alpha, HistoryOp};
use buvc_rs::indexer::Indexer;
use buvc_rs::journal::append_line;
use buvc_rs::snapshot_vals;
use buvc_rs::srs::{load_or_create_srs, make_ctx};
use buvc_rs::types::{JournalLine, SnapshotOut, UserState};
use buvc_rs::proof_server::ProofServerState;

/* ------------------------------------------------------------- */
/* Timing + metrics                                              */
/* ------------------------------------------------------------- */

fn t_start() -> Instant {
    Instant::now()
}
fn t_us(t0: Instant) -> u128 {
    t0.elapsed().as_micros() as u128
}

#[derive(serde::Serialize)]
struct Metric {
    phase: &'static str,
    block: u64,
    n: usize,
    alpha: usize,
    beta: usize,
    micros: u128,
}

fn emit(
    phase: &'static str,
    block: u64,
    n: usize,
    alpha: usize,
    beta: usize,
    micros: u128,
) {
    let m = Metric { phase, block, n, alpha, beta, micros };
    println!("METRIC {}", serde_json::to_string(&m).unwrap());
}

/* ------------------------------------------------------------- */
/* Helpers                                                       */
/* ------------------------------------------------------------- */

fn canonicalize_beta_delta(beta: &[usize], delta: &[Fr]) -> (Vec<usize>, Vec<Fr>) {
    let mut acc: BTreeMap<usize, Fr> = BTreeMap::new();
    for (&i, &d) in beta.iter().zip(delta.iter()) {
        if !d.is_zero() {
            *acc.entry(i).or_insert(Fr::ZERO) += d;
        }
    }
    acc.into_iter().unzip()
}

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

fn build_alpha_strict(
    alpha_addrs: &[Address],
    indexer: &Indexer,
) -> Result<(Vec<usize>, Vec<String>)> {
    if alpha_addrs.is_empty() {
        bail!("|α| must be > 0");
    }

    let mut tmp: Vec<(usize, Address)> = Vec::new();
    for &a in alpha_addrs {
        tmp.push((indexer.index_of(a)?, a));
    }

    tmp.sort_by_key(|(i, _)| *i);
    for w in tmp.windows(2) {
        if w[0].0 == w[1].0 {
            bail!("duplicate α index {}", w[0].0);
        }
    }

    let indices = tmp.iter().map(|(i, _)| *i).collect();
    let hex = tmp.iter().map(|(_, a)| format!("{:#x}", a)).collect();

    Ok((indices, hex))
}

/* ------------------------------------------------------------- */
/* CLI                                                           */
/* ------------------------------------------------------------- */

#[derive(Parser, Debug)]
#[command(name = "cauchy-runner")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    BuildUniverse {
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long)]
        start_block: u64,
        #[arg(long)]
        end_block: u64,
        #[arg(long)]
        out: PathBuf,
    },

    PublisherSnapshot {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        block: u64,
        #[arg(long)]
        universe_file: PathBuf,
        #[arg(long)]
        snapshot_vals: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },

    PublisherAdvance {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long)]
        universe_file: PathBuf,
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        snapshot_vals: PathBuf,
        #[arg(long)]
        end_block: u64,
        #[arg(long)]
        journal: PathBuf,
    },

    IssueUserState {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        universe_file: PathBuf,
        #[arg(long)]
        snapshot_vals: PathBuf,
        #[arg(long)]
        alpha_addresses: Option<String>,
        #[arg(long)]
        alpha_file: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },

    ProofServerInit {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        user_states: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },

    ProofServerAdvance {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        proof_server_state: PathBuf,
        #[arg(long)]
        journal: PathBuf,
        #[arg(long)]
        end_block: u64,
        #[arg(long)]
        out: PathBuf,
    },

    ProofServerHistory {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        proof_server_state: PathBuf,
        #[arg(long)]
        journal: PathBuf,
        #[arg(long)]
        start_block: u64,
        #[arg(long)]
        blocks: Vec<u64>,
        #[arg(long)]
        user_id: String,
        #[arg(long)]
        out: PathBuf,
    },

    UserVerify {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        history: PathBuf,
    },
}

fn main() -> Result<()> {
    color_eyre::install().ok();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::BuildUniverse {
            dataset_dir,
            start_block,
            end_block,
            out,
        } => cmd_build_universe(
            dataset_dir,
            start_block,
            end_block,
            out,
        ),

        Cmd::PublisherSnapshot {
            logn,
            srs,
            block,
            universe_file,
            snapshot_vals,
            out,
        } => cmd_publisher_snapshot(logn, srs, block, universe_file, snapshot_vals, out),

        Cmd::PublisherAdvance {
            logn,
            srs,
            dataset_dir,
            universe_file,
            snapshot,
            snapshot_vals,
            end_block,
            journal,
        } => cmd_publisher_advance(
            logn,
            srs,
            dataset_dir,
            universe_file,
            snapshot,
            snapshot_vals,
            end_block,
            journal,
        ),

        Cmd::IssueUserState {
            logn,
            srs,
            snapshot,
            universe_file,
            snapshot_vals,
            alpha_addresses,
            alpha_file,
            out,
        } => cmd_issue_user_state(
            logn,
            srs,
            snapshot,
            universe_file,
            snapshot_vals,
            alpha_addresses,
            alpha_file,
            out,
        ),

        Cmd::ProofServerInit {
            logn,
            srs,
            snapshot,
            user_states,
            out,
        } => cmd_proof_server_init(logn, srs, snapshot, user_states, out),

        Cmd::ProofServerAdvance {
            logn,
            srs,
            proof_server_state,
            journal,
            end_block,
            out,
        } => cmd_proof_server_advance(logn, srs, proof_server_state, journal, end_block, out),

        Cmd::ProofServerHistory {
                logn,
                srs,
                proof_server_state,
                journal,
                start_block,
                blocks,
                user_id,
                out,
            } => cmd_proof_server_history(
                logn,
                srs,
                proof_server_state,
                journal,
                start_block,
                blocks,
                user_id,
                out,
            ),

        Cmd::UserVerify {
            logn,
            srs,
            history,
        } => cmd_user_verify(logn, srs, history),
    }
}

/* ------------------------------------------------------------- */
/* Build universe                                                 */
/* ------------------------------------------------------------- */

fn cmd_build_universe(
    dataset_dir: PathBuf,
    start_block: u64,
    end_block: u64,
    out: PathBuf,
) -> Result<()> {
    let mut dataset = buvc_rs::dataset::DatasetReader::new(&dataset_dir, 100_000);
    let mut addrs = HashSet::<Address>::new();

    dataset.iterate_range(start_block as u32, end_block as u32, |_, entries| {
        for e in entries {
            addrs.insert(e.address);
        }
        Ok(())
    })?;

    let mut v: Vec<_> = addrs.into_iter().collect();
    v.sort();

    fs::write(
        &out,
        v.iter().map(|a| format!("{:#x}\n", a)).collect::<String>(),
    )?;

    eprintln!("Universe built: {} addresses", v.len());
    Ok(())
}

/* ------------------------------------------------------------- */
/* Publisher snapshot                                             */
/* ------------------------------------------------------------- */

fn cmd_publisher_snapshot(
    logn: usize,
    srs: PathBuf,
    block: u64,
    universe_file: PathBuf,
    snapshot_vals: PathBuf,
    out: PathBuf,
) -> Result<()> {
    let t0 = t_start();

    let n = 1usize << logn;
    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    let ctx = make_ctx(&vp, logn);

    let universe = read_addresses_file(&universe_file)?;
    let bals = snapshot_vals::read_u256hex_lines(&snapshot_vals)?;
    if universe.len() != bals.len() {
        bail!("snapshot_vals length mismatch");
    }

    let indexer = Indexer::from_universe_sequential(&universe, n)?;

    let mut v = vec![Fr::ZERO; n];
    for (a, bal) in universe.into_iter().zip(bals.into_iter()) {
        let i = indexer.index_of(a)?;
        v[i] = fr_from_u256_exact(bal)?;
    }

    let (gc, gq_full) = ctx.build_commitment(&v);


    let snap = SnapshotOut {
        block_number: block,
        n,
        logn,
        srs_id,
        gc_hex: g1_to_hex(&gc),
        values_hex: v.iter().map(fr_to_hex).collect(),
        witnesses_hex: gq_full.iter().map(g1_to_hex).collect(),
        universe_mode: "sequential".into(),
        balance_encoding: "fr_exact_from_u256".into(),
    };
    

    fs::write(&out, serde_json::to_vec_pretty(&snap)?)?;

    let micros = t_us(t0);
    let sz = fs::metadata(&out)?.len() as usize;
    // alpha=0, beta used as "bytes"
    emit(
        "[PublisherSnapshot]",
        block,
        n,
        0,
        sz,
        micros,
    );

    Ok(())
}


/* ------------------------------------------------------------- */
/* Publisher advance                                              */
/* ------------------------------------------------------------- */

fn cmd_publisher_advance(
    logn: usize,
    srs: PathBuf,
    dataset_dir: PathBuf,
    universe_file: PathBuf,
    snapshot: PathBuf,
    snapshot_vals: PathBuf,
    end_block: u64,
    journal: PathBuf,
) -> Result<()> {
    let n = 1usize << logn;

    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    let ctx = make_ctx(&vp, logn);

    let snap: SnapshotOut = serde_json::from_slice(&fs::read(snapshot)?)?;
    let universe = read_addresses_file(&universe_file)?;
    let indexer = Indexer::from_universe_sequential(&universe, n)?;

    let mut tracker = StateTracker::from_snapshot_vals_file(
        &dataset_dir,
        100_000,
        snap.block_number,
        &universe,
        &snapshot_vals,
    )?;

    let mut gc = g1_from_hex(&snap.gc_hex)?;
    let mut cur = snap.block_number;

    while cur < end_block {
        let t0 = t_start();

        let changes = tracker.apply_next_block()?;
        cur = tracker.cur_block;

        let mut beta = Vec::new();
        let mut delta = Vec::new();

        for (addr, old, new) in changes {
            let i = indexer.index_of(addr)?;
            beta.push(i);
            delta.push(delta_fr(new, old));
        }

        let (beta, delta) = canonicalize_beta_delta(&beta, &delta);

        let t1 = t_start();
        for (&i, &d) in beta.iter().zip(delta.iter()) {
            gc = ctx.update_commitment(gc, i, d);
        }
        let t2 = t_us(t1);

        append_line(
            &journal,
            &JournalLine {
                block_number: cur,
                n,
                logn,
                srs_id: srs_id.clone(),
                changed_indices: beta,
                delta_hex: delta.iter().map(fr_to_hex).collect(),
                gc_hex: Some(g1_to_hex(&gc)),
            },
        )?;

        emit(
            "[PublisherAdvance] block",
            cur,
            n,
            0,
            delta.len(),
            t_us(t0),
        );
        emit(
            "[CommitUpdate] block",
            cur,
            n,
            0,
            delta.len(),
            t2,
        );
    }

    Ok(())
}

/* ------------------------------------------------------------- */
/* Issue user state                                               */
/* ------------------------------------------------------------- */

fn cmd_issue_user_state(
    logn: usize,
    srs: PathBuf,
    snapshot: PathBuf,
    universe_file: PathBuf,
    _snapshot_vals: PathBuf,
    alpha_addresses: Option<String>,
    alpha_file: Option<PathBuf>,
    out: PathBuf,
) -> Result<()> {
    let t0 = t_start();
    let n = 1usize << logn;

    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    let _ctx = make_ctx(&vp, logn);

    let snap: SnapshotOut = serde_json::from_slice(&fs::read(snapshot)?)?;
    let universe = read_addresses_file(&universe_file)?;
    let indexer = Indexer::from_universe_sequential(&universe, n)?;

    let alpha_addrs = if let Some(f) = alpha_file {
        read_addresses_file(&f)?
    } else {
        parse_addr_list(&alpha_addresses.unwrap())?
    };

    let (alpha_idx, alpha_hex) = build_alpha_strict(&alpha_addrs, &indexer)?;
    let alpha_len = alpha_idx.len();
    // Slice balances & witnesses from snapshot
    let vals = alpha_idx
    .iter()
    .map(|&i| snap.values_hex[i].clone())
    .collect();

    let wit = alpha_idx
    .iter()
    .map(|&i| snap.witnesses_hex[i].clone())
    .collect::<Vec<_>>();

    let user = UserState {
        n,
        logn,
        srs_id,
        last_block: snap.block_number,
        gc_hex: snap.gc_hex,
        alpha_indices: alpha_idx,
        alpha_addresses: alpha_hex,
        alpha_values_hex: vals,
        alpha_witnesses_hex: wit,
        universe_mode: snap.universe_mode,
        value_mode: "snapshot".to_string(),
        witness_mode: "snapshot".to_string(),
    };
    
    

    fs::write(&out, serde_json::to_vec_pretty(&user)?)?;
    let micros = t_us(t0);
    let sz = fs::metadata(&out)?.len() as usize;
    emit(
        "[IssueUserState]",
        snap.block_number,
        n,
        alpha_len, // use stored len
        sz,
        micros,
    );
    Ok(())
}    
/* ------------------------------------------------------------- */
/* Proof-server init + advance                                    */
/* ------------------------------------------------------------- */

fn cmd_proof_server_init(
    logn: usize,
    srs: PathBuf,
    snapshot: PathBuf,
    user_states: Vec<PathBuf>,
    out: PathBuf,
) -> Result<()> {
    let t0 = t_start();
    // Ensure SRS exists and capture its id
    let (_vp, srs_id) = load_or_create_srs(&srs, logn)?;
    let snap: SnapshotOut = serde_json::from_slice(&fs::read(&snapshot)?)?;

    if snap.srs_id != srs_id {
        bail!(
            "SRS id mismatch between snapshot ({}) and srs.bin ({})",
            snap.srs_id,
            srs_id
        );
    }

    let mut users = Vec::new();
    for (i, p) in user_states.iter().enumerate() {
        let u: UserState = serde_json::from_slice(&fs::read(p)?)?;
    
        // 🛡️ Sanity assertions per user
        assert_eq!(
            u.alpha_indices.len(),
            u.alpha_values_hex.len(),
            "alpha length mismatch in values for user {}",
            i + 1
        );
        assert_eq!(
            u.alpha_indices.len(),
            u.alpha_witnesses_hex.len(),
            "alpha length mismatch in witnesses for user {}",
            i + 1
        );
    
        users.push((format!("user{}", i + 1), u));
    }
    

    let ps = ProofServerState::from_user_states(&users)?;

    // sanity: align snapshot vs union user state
    if ps.n != snap.n || ps.logn != snap.logn || ps.gc_hex != snap.gc_hex {
        bail!("snapshot.json inconsistent with user_state files");
    }

    fs::write(&out, serde_json::to_vec_pretty(&ps)?)?;
    let micros = t_us(t0);
    let sz = fs::metadata(&out)?.len() as usize;
    emit(
        "[ProofServerInit]",
        ps.last_block,
        ps.n,
        ps.alpha_indices.len(), // union α
        sz,                     // bytes
        micros,
    );
    Ok(())
    
}

fn cmd_proof_server_advance(
    logn: usize,
    srs: PathBuf,
    proof_server_state: PathBuf,
    journal: PathBuf,
    end_block: u64,
    out: PathBuf,
) -> Result<()> {
    let mut ps: ProofServerState =
        serde_json::from_slice(&fs::read(&proof_server_state)?)?;

    let (vp, _) = load_or_create_srs(&srs, logn)?;
    let ctx = make_ctx(&vp, logn);

    // Take copies so we don't borrow `ps` immutably for the whole call.
    let start_block = ps.last_block;
    let srs_id = ps.srs_id.clone();
    let n = ps.n;
    let logn_ps = ps.logn;
    let mut blocks_applied = 0usize;
    let mut updates_applied = 0usize;
    let t_total = t_start();

    buvc_rs::journal::for_each_line_filtered(
        &journal,
        &srs_id,
        n,
        logn_ps,
        |j: JournalLine| -> Result<()> {
            if j.block_number <= start_block || j.block_number > end_block {
                return Ok(());
            }

            let t_block = t_start();
            let mut delta = Vec::<Fr>::new();
            for h in &j.delta_hex {
                delta.push(fr_from_hex(h)?);
            }

            let (beta, delta) =
                canonicalize_beta_delta(&j.changed_indices, &delta);

            ps.apply_update(
                &ctx,
                &beta,
                &delta,
                j.block_number,
                j.gc_hex.as_deref(),
            )?;
            blocks_applied += 1;
            updates_applied += beta.len();

            let micros = t_us(t_block);
            emit(
                "[ProofServerAdvanceBlock]",
                j.block_number,
                ps.n,
                ps.alpha_indices.len(),
                beta.len(),
                micros,
            );
            
            eprintln!(
                "[ProofServerAdvance] block={} beta={} micros={}",
                j.block_number,
                beta.len(),
                micros
            );
            Ok(())
        },
    )?;

    fs::write(out, serde_json::to_vec_pretty(&ps)?)?;
    let total_micros = t_us(t_total);
    emit(
        "[ProofServerAdvanceTotal]",
        ps.last_block,
        ps.n,
        ps.alpha_indices.len(),
        updates_applied,
        total_micros,
    );
    eprintln!(
        "[ProofServerAdvance] blocks_applied={} updates={} total_micros={}",
        blocks_applied,
        updates_applied,
        total_micros
    );
    Ok(())
}


/* ------------------------------------------------------------- */
/* History output structure                                       */
/* ------------------------------------------------------------- */

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ProofServerHistoryOut {
    pub user_id: String,
    pub block: u64,
    pub indices: Vec<usize>,
    pub values_hex: Vec<String>,
    pub witnesses_hex: Vec<String>,
    pub commitment_hex: String,
}

/* ------------------------------------------------------------- */
/* Proof-server history query                                     */
/* ------------------------------------------------------------- */

fn cmd_proof_server_history(
    logn: usize,
    srs: PathBuf,
    proof_server_state: PathBuf,
    journal: PathBuf,
    start_block: u64,
    mut blocks: Vec<u64>,
    user_id: String,
    out: PathBuf,
) -> Result<()> {
    let t_total = t_start();
    // ---- Load proof-server state ----
    let ps: ProofServerState =
        serde_json::from_slice(&fs::read(&proof_server_state)?)?;
        assert_eq!(ps.alpha_indices.len(), ps.alpha_witnesses_hex.len());

    if ps.logn != logn {
        bail!("logn mismatch");
    }

    // ---- Load SRS + context ----
    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    if ps.srs_id != srs_id {
        bail!("srs_id mismatch");
    }
    let ctx = make_ctx(&vp, logn);

    // ---- Validate user ----
    let (user_alpha, user_pos) = ps
        .user_alpha(&user_id)
        .ok_or_else(|| eyre::eyre!("unknown user_id {}", user_id))?;

    if blocks.is_empty() {
        bail!("--blocks must be non-empty");
    }
    blocks.sort_unstable();
    blocks.dedup();

    if blocks[0] <= start_block {
        bail!("query block <= start_block");
    }
    if *blocks.last().unwrap() > ps.last_block {
        bail!("query block beyond server head");
    }

    // ---- Final witnesses + values (at last_block) ----
    let mut gq_final = Vec::<G1>::new();
    for hx in &ps.alpha_witnesses_hex {
        gq_final.push(g1_from_hex(hx)?);
    }

    let mut vals_final = Vec::<Fr>::new();
    for hx in &ps.alpha_values_hex {
        vals_final.push(fr_from_hex(hx)?);
    }

    // ---- Build forward history ops ----
    let mut ops_fwd = Vec::<HistoryOp>::new();
    let mut qid_to_block = Vec::<u64>::new();
    let mut qpos = 0usize;
    let _blocks_scanned = 0usize;
    let _updates_scanned = 0usize;

    let t_journal = t_start();
    buvc_rs::journal::for_each_line_filtered(
        &journal,
        &ps.srs_id,
        ps.n,
        ps.logn,
        |j: JournalLine| -> Result<()> {
            let b = j.block_number;
            let max_q = *blocks.last().unwrap();
            if j.block_number <= start_block || j.block_number > max_q {
                return Ok(());
            }
            

            // UPDATE
            let mut delta = Vec::<Fr>::new();
            for hx in &j.delta_hex {
                delta.push(fr_from_hex(hx)?);
            }
            let (beta, delta) =
                canonicalize_beta_delta(&j.changed_indices, &delta);

            if !beta.is_empty() {
                ops_fwd.push(HistoryOp::Update { beta, delta });
            }

            // QUERY
            while qpos < blocks.len() && blocks[qpos] < b {
                qpos += 1;
            }
            if qpos < blocks.len() && blocks[qpos] == b {
                let qid = qid_to_block.len();
                ops_fwd.push(HistoryOp::Query { query_id: qid });
                qid_to_block.push(b);
                qpos += 1;
            }

            Ok(())
        },
    )?;
    let journal_micros = t_us(t_journal);
    emit(
        "[ProofServerHistoryJournal]",
        0,
        ps.n,
        ps.alpha_indices.len(),
        qid_to_block.len(),
        journal_micros,
    );
    eprintln!(
        "[ProofServerHistory] journal_scan queries={} micros={}",
        qid_to_block.len(),
        journal_micros
    );
    let t0 = t_start();
    // ---- Reverse ops + negate deltas ----
    let mut ops_rev = ops_fwd;
    ops_rev.reverse();
    for op in ops_rev.iter_mut() {
        if let HistoryOp::Update { delta, .. } = op {
            for d in delta.iter_mut() {
                *d = -*d;
            }
        }
    }

    // ---- Rewind witnesses (paper algorithm) ----
    let hist = vupdate_history_same_alpha(
        &ctx,
        &ps.alpha_indices,
        &gq_final,
        &ops_rev,
    );

    // ---- Rewind values (linear replay; application-layer) ----
    let mut vals_map: HashMap<u64, Vec<Fr>> = HashMap::new();
    let mut cur_vals = vals_final.clone();

    for op in ops_rev.iter() {
        match op {
            HistoryOp::Update { beta, delta } => {
                for (&b, &d) in beta.iter().zip(delta.iter()) {
                    if let Some(pos) = ps.alpha_indices.iter().position(|&i| i == b) {
                        cur_vals[pos] += d;
                    }
                }
            }
            HistoryOp::Query { query_id } => {
                vals_map.insert(qid_to_block[*query_id], cur_vals.clone());
            }
        }
    }

    // ---- Assemble output ----
    let mut out_all = Vec::<ProofServerHistoryOut>::new();

    for r in hist {
        let block = qid_to_block[r.query_id];
        let t_q = t_start();
        let all_vals = vals_map
            .get(&block)
            .ok_or_else(|| eyre::eyre!("missing values for block {}", block))?;

        let mut user_vals = Vec::new();
        let mut user_wits = Vec::new();

        for &p in &user_pos {
            user_vals.push(fr_to_hex(&all_vals[p]));
            user_wits.push(g1_to_hex(&r.proofs[p]));
        }
        let dt_q = t_us(t_q);
        emit(
            "[ProofServerHistoryPerQuery]",
            block,
            ps.n,
            user_pos.len(), // user α
            0,
            dt_q,
        );

        out_all.push(ProofServerHistoryOut {
            user_id: user_id.clone(),
            block,
            indices: user_alpha.clone(),
            values_hex: user_vals,
            witnesses_hex: user_wits,
            commitment_hex: ps.gc_hex.clone(),
        });
    }
    
    out_all.sort_by_key(|o| o.block);


    let micros = t_us(t0);
    // Core history algorithm latency (does NOT include state/SRS load or journal scan)
    emit(
        "[ProofServerHistoryCore]",
        0,                         // block field not meaningful here
        ps.n,                      // n = vector size
        ps.alpha_indices.len(),    // alpha = union α size
        out_all.len(),             // beta field reused as "queries" = #blocks
        micros,
    );
    eprintln!(
        "[ProofServerHistory] core queries={} micros={}",
        out_all.len(),
        micros
    );
 let json = serde_json::to_vec_pretty(&out_all)?;
    let payload_bytes = json.len() as usize;

    emit(
        "[ProofServerHistoryPayload]",
        0,
        ps.n,
        ps.alpha_indices.len(), // union α
        payload_bytes,          // β used as size
        0,
    );
    // ---- Write output ----
    fs::write(&out, serde_json::to_vec_pretty(&out_all)?)?;
    eprintln!(
        "[ProofServerHistory] user={} blocks={} -> {}",
        user_id,
        out_all.len(),
        out.display()
    );
    let total_micros = t_us(t_total);
    emit(
        "[ProofServerHistoryTotal]",
        0,
        ps.n,
        ps.alpha_indices.len(),
        out_all.len(),
        total_micros,
    );
    eprintln!(
        "[ProofServerHistory] total queries={} micros={}",
        out_all.len(),
        total_micros
    );
    Ok(())
}


/* ------------------------------------------------------------- */
/* User verification                                              */
/* ------------------------------------------------------------- */
fn cmd_user_verify(
    logn: usize,
    srs: PathBuf,
    history: PathBuf,
) -> Result<()> {
    // Load history produced by proof-server-history
    let hist: Vec<ProofServerHistoryOut> =
        serde_json::from_slice(&fs::read(&history)?)?;

    let (vp, _) = load_or_create_srs(&srs, logn)?;
    let ctx = make_ctx(&vp, logn);

    // n = vector size (for metrics)
    let n = 1usize << logn;

    // ---- totals for [UserVerifyTotal] ----
    let mut total_alpha = 0usize;
    let mut proofs = 0usize;
    let t_total = t_start();

    for h in hist {
        let gc = g1_from_hex(&h.commitment_hex)?;

        let mut vals = Vec::<Fr>::new();
        let mut gqs = Vec::<G1>::new();

        for (v_hex, w_hex) in h.values_hex.iter().zip(h.witnesses_hex.iter()) {
            vals.push(fr_from_hex(v_hex)?);
            gqs.push(g1_from_hex(w_hex)?);
        }

        let t0 = t_start();
        let agg = ctx.aggregate_proof(&h.indices, &gqs);
        let ok = ctx.verify_multi(&vp, gc, &h.indices, &vals, agg);
        let dt = t_us(t0);

        emit(
            "[UserVerify]",
            h.block,
            n,
            h.indices.len(), // alpha = #indices checked in this proof
            0,               // beta not meaningful here
            dt,
        );

        println!(
            "user={} block={} verify_multi={} micros={}",
            h.user_id, h.block, ok, dt
        );

        // accumulate totals
        total_alpha += h.indices.len();
        proofs += 1;
    }

    let total_micros = t_us(t_total);
    emit(
        "[UserVerifyTotal]",
        0,
        n,
        total_alpha, // total α checked across all proofs
        proofs,      // β reused as "#proofs"
        total_micros,
    );

    Ok(())
}
