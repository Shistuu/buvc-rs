use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::{Field, Zero};

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{BufRead, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;

use ethers_core::types::Address;

use eyre::{bail, Result};

use buvc_rs::{
    codec::{
        fr_from_hex,fr_from_u256_exact, fr_to_hex, g1_from_hex, g1_to_hex, delta_fr,
    },
    snapshot_vals,
    StateTracker,
    Indexer,
    journal::append_line,
    srs::{load_or_create_srs, make_ctx},
    JournalLine, SnapshotOut, UserState,
    ProofServerState,
    types::{Metric, HistoryOp, Cli, Cmd, JournalLineLite, ProofServerHistoryOut, ValueClaim},
    vc_context::VcContext,
    history::vupdate_history_vupdate_dc,
    vc_parameter::VcParameter
};
use std::sync::OnceLock;

struct LoadedCtx {
    logn: usize,
    srs_path: PathBuf,
    vp: VcParameter,
    srs_id: String,
    ctx: VcContext,
}


fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

static LOADED: OnceLock<LoadedCtx> = OnceLock::new();

fn get_loaded_ctx(logn: usize, srs: &Path) -> Result<&'static LoadedCtx> {
    if let Some(lc) = LOADED.get() {
        // Enforce: same srs + same logn inside one daemon lifetime
        if lc.logn != logn {
            bail!("Loaded ctx logn={} but requested logn={}", lc.logn, logn);
        }
        if lc.srs_path.as_path() != srs {
            bail!(
                "Loaded ctx srs_path={:?} but requested srs_path={:?}",
                lc.srs_path,
                srs
            );
        }
        return Ok(lc);
    }

    // First time in this process: load + build ctx once
    let t0 = t_start();
    let (vp, srs_id) = load_or_create_srs(srs, logn)?;
    let ctx = make_ctx(&vp, logn);
    let _ = LOADED.set(LoadedCtx {
        logn,
        srs_path: srs.to_path_buf(),
        vp,
        srs_id,
        ctx,
    });

    let n = 1usize << logn;
    emit("[InitSRS+CtxOnce]", 0, n, 0, 0, t_us(t0));

    Ok(LOADED.get().unwrap())
}

fn t_start() -> Instant {
    Instant::now()
}
fn t_us(t0: Instant) -> u128 {
    t0.elapsed().as_micros() as u128
}

/// Emit metrics to stdout
fn emit(phase: &'static str, block: u64, n: usize, alpha: usize, beta: usize, micros: u128) {
    let m = Metric { phase, block, n, alpha, beta, micros };
    println!("METRIC {}", serde_json::to_string(&m).unwrap());
}

/// Emit metrics to an arbitrary writer (per-request log file)
fn emit_to<W: Write>(
    w: &mut W,
    phase: &'static str,
    block: u64,
    n: usize,
    alpha: usize,
    beta: usize,
    micros: u128,
) {
    let m = Metric { phase, block, n, alpha, beta, micros };
    let _ = writeln!(w, "METRIC {}", serde_json::to_string(&m).unwrap());
}

fn log_line<W: Write>(w: &mut W, s: &str) {
    let _ = writeln!(w, "{}", s);
}

fn canonicalize_beta_delta(beta: &[usize], delta: &[Fr]) -> (Vec<usize>, Vec<Fr>) {
    let mut acc: BTreeMap<usize, Fr> = BTreeMap::new();
    for (&i, &d) in beta.iter().zip(delta.iter()) {
        if !d.is_zero() {
            *acc.entry(i).or_insert(Fr::ZERO) += d;
        }
    }
    // drop any entries that summed to 0
    acc.retain(|_, v| !v.is_zero());

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

fn main() -> Result<()> {
    color_eyre::install().ok();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::BuildUniverse { dataset_dir, start_block, end_block, out } => {
            cmd_build_universe(dataset_dir, start_block, end_block, out)?
        }

        Cmd::PublisherSnapshot { logn, srs, block, universe_file, snapshot_vals, out } => {
            cmd_publisher_snapshot(logn, srs, block, universe_file, snapshot_vals, out)?
        }

        Cmd::PublisherAdvance {
            logn,
            srs,
            dataset_dir,
            universe_file,
            snapshot,
            snapshot_vals,
            end_block,
            journal,
            alpha_addresses,
            proof_server_out,
            block_log_csv,
            cancel_path,
        } => {
            cmd_publisher_advance(
                logn,
                srs,
                dataset_dir,
                universe_file,
                snapshot,
                snapshot_vals,
                end_block,
                journal,
                alpha_addresses,
                proof_server_out,
                block_log_csv,
                cancel_path,
            )?
        }

        Cmd::IssueUserState {
            logn,
            srs,
            snapshot,
            universe_file,
            addresses,
            snapshot_vals,
            out,
        } => {
            cmd_issue_user_state(
                logn,
                srs,
                snapshot,
                universe_file,
                addresses,
                snapshot_vals,
                out,
            )?
        }

        Cmd::ProofServerInit { logn, srs, snapshot, user_id, user_state, out } => {
            cmd_proof_server_init(logn, srs, snapshot, user_id, user_state, out)?
        }

        Cmd::ProofServerAdvance { logn, srs, proof_server_state, journal, end_block, out } => {
            cmd_proof_server_advance(logn, srs, proof_server_state, journal, end_block, out)?
        }

        Cmd::ProofServerHistory { logn, srs, proof_server_state, journal, start_block, blocks, user_id, out } => {
            cmd_proof_server_history(logn, srs, proof_server_state, journal, start_block, blocks, user_id, out)?
        }

        Cmd::ProofServerHistoryServer { logn, srs, proof_server_state, journal, r#in } => {
            cmd_proof_server_history_server(logn, srs, proof_server_state, journal, r#in)?
        }

        Cmd::ExtractValue {
            dataset_dir,
            snapshot,
            snapshot_vals,
            universe_file,
            target_block,
            index,
        } => {
            cmd_extract_value(
                dataset_dir,
                snapshot,
                snapshot_vals,
                universe_file,
                target_block,
                index,
            )?
        }

        Cmd::UserVerify { logn, srs, user_state, history, claims } => {
            cmd_user_verify(logn, srs, user_state, history, claims)?
        }
        Cmd::Daemon { logn, srs, r#in } => {
            cmd_daemon(logn, srs, r#in)?
        }
    }

    Ok(())
}



/// Build universe of addresses from dataset blocks
fn cmd_build_universe(dataset_dir: PathBuf, start_block: u64, end_block: u64, out: PathBuf) -> Result<()> {
    let mut dataset = buvc_rs::DatasetReader::new(dataset_dir, 100_000);
    let mut addrs = HashSet::<Address>::new();

    dataset.iterate_range(start_block as u32, end_block as u32, |_, entries| {
        for e in entries {
            addrs.insert(e.address);
        }
        Ok(())
    })?;

    let mut v: Vec<_> = addrs.into_iter().collect();
    v.sort();

    fs::write(&out, v.iter().map(|a| format!("{:#x}\n", a)).collect::<String>())?;
    eprintln!("Universe built: {} addresses", v.len());
    Ok(())
}

// /// Create initial publisher snapshot at given block
// fn cmd_publisher_snapshot(
//     logn: usize,
//     srs: PathBuf,
//     block: u64,
//     universe_file: PathBuf,
//     snapshot_vals_path: PathBuf,
//     out: PathBuf,
// ) -> Result<()> {
//     let t0 = t_start();

//     let n = 1usize << logn;
//     let lc = get_loaded_ctx(logn, &srs)?;
//     let ctx = &lc.ctx;
//     let srs_id = lc.srs_id.clone();

//     let universe = read_addresses_file(&universe_file)?;
//     let bals = snapshot_vals::read_u256hex_lines(&snapshot_vals_path)?;
//     if universe.len() != bals.len() {
//         bail!("snapshot_vals length mismatch");
//     }

//     let indexer = Indexer::from_universe_sequential(&universe, n)?;

//     let mut v = vec![Fr::ZERO; n];
//     for (a, bal) in universe.into_iter().zip(bals.into_iter()) {
//         let i = indexer.index_of(a)?;
//         v[i] = fr_from_u256_exact(bal)?;
//     }

//     let (gc, _gq_full) = ctx.build_commitment(&v);

//     let snap = SnapshotOut {
//         block_number: block,
//         n,
//         logn,
//         srs_id,
//         gc_hex: g1_to_hex(&gc),
//         universe_mode: "sequential".into(),
//         balance_encoding: "fr_exact_from_u256".into(),
//     };

//     fs::write(&out, serde_json::to_vec_pretty(&snap)?)?;

//     let micros = t_us(t0);
//     let sz = fs::metadata(&out)?.len() as usize;
//     emit("[PublisherSnapshot]", block, n, 0, sz, micros);
//     Ok(())
// }
/// Create initial publisher snapshot at given block
fn cmd_publisher_snapshot(
    logn: usize,
    srs: PathBuf,
    block: u64,
    universe_file: PathBuf,
    snapshot_vals_path: PathBuf,
    out: PathBuf,
) -> Result<()> {
    // Total (includes any init done inside this function)
    let t_total = t_start();

    let n = 1usize << logn;

    // ---- ctx lookup (may trigger InitSRS+CtxOnce)
    // Measure how long the ctx lookup/init took inside THIS call.
    let t_ctx = t_start();
    let lc = get_loaded_ctx(logn, &srs)?;
    let ctx_us = t_us(t_ctx);

    let ctx = &lc.ctx;
    let srs_id = lc.srs_id.clone();

    // From here on, this is the “snapshot work” excluding ctx init.
    let t_snap_only = t_start();

    let universe = read_addresses_file(&universe_file)?;
    let universe_len = universe.len();

    let bals = snapshot_vals::read_u256hex_lines(&snapshot_vals_path)?;
    if universe_len != bals.len() {
        bail!("snapshot_vals length mismatch");
    }

    let indexer = Indexer::from_universe_sequential(&universe, n)?;

    let mut v = vec![Fr::ZERO; n];
    for (a, bal) in universe.into_iter().zip(bals.into_iter()) {
        let i = indexer.index_of(a)?;
        v[i] = fr_from_u256_exact(bal)?;
    }

    let (gc, _gq_full) = ctx.build_commitment(&v);

    let snap = SnapshotOut {
        block_number: block,
        n,
        logn,
        srs_id,
        gc_hex: g1_to_hex(&gc),
        universe_mode: "sequential".into(),
        balance_encoding: "fr_exact_from_u256".into(),
    };

    fs::write(&out, serde_json::to_vec_pretty(&snap)?)?;

    // ---- metrics
    let snap_only_us = t_us(t_snap_only);
    let total_us = t_us(t_total);
    let out_sz = fs::metadata(&out)?.len() as usize;

    // This is the number you want for “snapshot generation time only”
    emit("[PublisherSnapshotOnly]", block, n, 0, universe_len, snap_only_us);

    // Optional: how long ctx lookup/init took inside this call
    emit("[PublisherSnapshotCtxLookup]", block, n, 0, 0, ctx_us);

    // Total end-to-end time (includes ctx lookup/init)
    emit("[PublisherSnapshotTotal]", block, n, 0, out_sz, total_us);

    // If you want to keep your old label too (optional; same as Total)
    // emit("[PublisherSnapshot]", block, n, 0, out_sz, total_us);

    Ok(())
}

/// Advance publisher snapshot to new block range
fn cmd_publisher_advance(
    logn: usize,
    srs: PathBuf,
    dataset_dir: PathBuf,
    universe_file: PathBuf,
    snapshot: PathBuf,
    snapshot_vals_path: PathBuf,
    end_block: u64,
    journal: PathBuf,

    // NEW
    alpha_addresses: Option<PathBuf>,
    // NEW
    proof_server_out: Option<PathBuf>,
    block_log_csv: Option<PathBuf>,
    cancel_path: Option<PathBuf>,
) -> Result<()> {
    let n = 1usize << logn;

    let lc = get_loaded_ctx(logn, &srs)?;
    let ctx = &lc.ctx;
    let srs_id = lc.srs_id.clone();

    let snap: SnapshotOut = serde_json::from_slice(&fs::read(&snapshot)?)?;
    if snap.logn != logn || snap.srs_id != srs_id {
        bail!("snapshot / srs mismatch");
    }

    let universe = read_addresses_file(&universe_file)?;
    let indexer = Indexer::from_universe_sequential(&universe, n)?;

    let mut tracker = StateTracker::from_snapshot_vals_file(
        &dataset_dir,
        100_000,
        snap.block_number,
        &universe,
        &snapshot_vals_path,
    )?;

    // publisher head commitment
    let mut gc = g1_from_hex(&snap.gc_hex)?;
    let mut cur = snap.block_number;
    let mut blocks_processed: usize = 0;
    let mut total_commit_us: u128 = 0;
    let mut total_witness_us: u128 = 0;
    let mut total_proc_us: u128 = 0;
    let mut block_log: Option<std::fs::File> = match &block_log_csv {
        Some(p) => {
            let mut f = std::fs::File::create(p)?;
            writeln!(f, "kind,block,start_ms,end_ms,wall_ms,proc_micros,beta")?;
            Some(f)
        }
        None => None,
    };
    
    // helper to write marker rows
    let mut log_marker = |kind: &str| {
        if let Some(f) = block_log.as_mut() {
            let t = now_ms();
            let _ = writeln!(f, "{},0,{},{},0,0,0", kind, t, t);
        }
    };
    let mut ps_opt: Option<(ProofServerState, Vec<G1>)> = None;
    
    if let Some(alpha_path) = alpha_addresses.as_ref() {
        // marker: alpha init starts (BEFORE expensive init)
        log_marker("init_alpha_start");
    
        // Build alpha_indices from alpha.txt addresses
        let addrs = read_addresses_file(alpha_path)?;
        if addrs.is_empty() {
            bail!("alpha_addresses file is empty");
        }
    
        let mut set = std::collections::BTreeSet::new();
        for a in addrs {
            let i = indexer.index_of(a)?;
            set.insert(i);
        }
        let alpha_indices: Vec<usize> = set.into_iter().collect();
        let alpha_len = alpha_indices.len();
    
        // Build initial witnesses at snapshot
        let bals = snapshot_vals::read_u256hex_lines(&snapshot_vals_path)?;
        if bals.len() != universe.len() {
            bail!("snapshot_vals length mismatch");
        }
    
        let mut v = vec![Fr::ZERO; n];
        for (a, bal) in universe.iter().zip(bals.into_iter()) {
            let i = indexer.index_of(*a)?;
            v[i] = fr_from_u256_exact(bal)?;
        }
    
        let t_init = t_start();
        let (gc_rebuilt, gq_alpha) = ctx.build_commitment_for_alpha(&v, &alpha_indices);
    
        if g1_to_hex(&gc_rebuilt) != snap.gc_hex {
            bail!("rebuilt snapshot commitment mismatch (alpha init)");
        }
    
        emit("[PublisherOnePassInitAlpha]", snap.block_number, n, alpha_len, 0, t_us(t_init));
    
        // marker: alpha init ends (AFTER init finished)
        log_marker("init_alpha_end");
    
        let ps = ProofServerState {
            n,
            logn,
            srs_id: srs_id.clone(),
            last_block: snap.block_number,
            gc_hex: snap.gc_hex.clone(),
            alpha_indices,
            alpha_witnesses_hex: gq_alpha.iter().map(g1_to_hex).collect(),
            users: vec![],
        };
    
        ps_opt = Some((ps, gq_alpha));
    }
    // let mut ps_opt: Option<(ProofServerState, Vec<G1>)> = None;

    // if let Some(alpha_path) = alpha_addresses.as_ref() {
    //     // Build alpha_indices from alpha.txt addresses
    //     let addrs = read_addresses_file(alpha_path)?;
    //     if addrs.is_empty() {
    //         bail!("alpha_addresses file is empty");
    //     }

    //     let mut set = std::collections::BTreeSet::new();
    //     for a in addrs {
    //         let i = indexer.index_of(a)?;
    //         set.insert(i);
    //     }
    //     let alpha_indices: Vec<usize> = set.into_iter().collect();
    //     let alpha_len = alpha_indices.len();

    //     // Build initial witnesses at snapshot using SAME method as IssueUserState
    //     let bals = snapshot_vals::read_u256hex_lines(&snapshot_vals_path)?;
    //     if bals.len() != universe.len() {
    //         bail!("snapshot_vals length mismatch");
    //     }

    //     let mut v = vec![Fr::ZERO; n];
    //     for (a, bal) in universe.iter().zip(bals.into_iter()) {
    //         let i = indexer.index_of(*a)?;
    //         v[i] = fr_from_u256_exact(bal)?;
    //     }

    //     let t_init = t_start();
    //     let (gc_rebuilt, gq_alpha) = ctx.build_commitment_for_alpha(&v, &alpha_indices);

    //     if g1_to_hex(&gc_rebuilt) != snap.gc_hex {
    //         bail!("rebuilt snapshot commitment mismatch (alpha init)");
    //     }
    //     let mut block_log: Option<std::fs::File> = match &block_log_csv {
    //         Some(p) => {
    //             let mut f = std::fs::File::create(p)?;
    //             // NEW: add a "kind" column + keep everything else
    //             writeln!(f, "kind,block,start_ms,end_ms,wall_ms,proc_micros,beta")?;
    //             Some(f)
    //         }
    //         None => None,
    //     };
        
    //     log_marker("init_alpha_start", now_ms());

    //     emit("[PublisherOnePassInitAlpha]", snap.block_number, n, alpha_len, 0, t_us(t_init));
    //     log_marker("init_alpha_end", now_ms());

    //     let ps = ProofServerState {
    //         n,
    //         logn,
    //         srs_id: srs_id.clone(),
    //         last_block: snap.block_number,
    //         gc_hex: snap.gc_hex.clone(),
    //         alpha_indices,
    //         alpha_witnesses_hex: gq_alpha.iter().map(g1_to_hex).collect(),
    //         users: vec![], // not needed for your one-pass maintenance
    //     };

    //     ps_opt = Some((ps, gq_alpha));
    // }

    while cur < end_block {
        if cancelled(cancel_path.as_deref()) {
            eprintln!("publisher_advance cancelled at block {}", cur);
            break;
        }
    
        let start_ms = now_ms();
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
    
        // commitment update
        let t1 = t_start();
        for (&i, &d) in beta.iter().zip(delta.iter()) {
            gc = ctx.update_commitment(gc, i, d);
        }
        let commit_us = t_us(t1);
        total_commit_us += commit_us;
    
        // journal append
        append_line(
            &journal,
            &JournalLine {
                block_number: cur,
                n,
                logn,
                srs_id: srs_id.clone(),
                changed_indices: beta.clone(),
                delta_hex: delta.iter().map(fr_to_hex).collect(),
                gc_hex: Some(g1_to_hex(&gc)),
            },
        )?;
    
        // witness update (optional)
        if let Some((ps, gq)) = ps_opt.as_mut() {
            let t_wit = t_start();
            if !beta.is_empty() && !ps.alpha_indices.is_empty() {
                *gq = ctx.update_witnesses_batch(&ps.alpha_indices, gq, &beta, &delta);
            }
            let wit_us = t_us(t_wit);
            total_witness_us += wit_us;
    
            ps.last_block = cur;
            ps.gc_hex = g1_to_hex(&gc);
            ps.alpha_witnesses_hex = gq.iter().map(g1_to_hex).collect();
    
            emit("[PublisherOnePassWitnessOnly] block", cur, n, ps.alpha_indices.len(), beta.len(), wit_us);
        }
    
        // end-to-end timing
        let proc_micros = t_us(t0);
        total_proc_us += proc_micros;
        blocks_processed += 1;
        let end_ms = now_ms();
        let wall_ms = end_ms - start_ms;
    
        // keep your existing metrics
        emit("[PublisherAdvance] block", cur, n, 0, beta.len(), proc_micros);
        emit("[CommitUpdate] block", cur, n, 0, beta.len(), commit_us);
    
        // CSV log
        if let Some(f) = block_log.as_mut() {
            let _ = writeln!(
                f,
                "block,{},{},{},{},{},{}",
                cur, start_ms, end_ms, wall_ms, proc_micros, beta.len()
            );
        }
    }

    // -----------------------------
    // Write final proof server state (latest witnesses only)
    // -----------------------------
    if let (Some(out_path), Some((ps, _gq))) = (proof_server_out, ps_opt.as_ref()) {
        fs::write(out_path, serde_json::to_vec_pretty(ps)?)?;
    }
    emit(
    "[PublisherAdvanceCommitTotal]",
    cur,
    n,
    ps_opt.as_ref().map(|(ps, _)| ps.alpha_indices.len()).unwrap_or(0),
    blocks_processed,
    total_commit_us,
);

emit(
    "[PublisherAdvanceWitnessTotal]",
    cur,
    n,
    ps_opt.as_ref().map(|(ps, _)| ps.alpha_indices.len()).unwrap_or(0),
    blocks_processed,
    total_witness_us,
);

emit(
    "[PublisherAdvanceProcTotal]",
    cur,
    n,
    ps_opt.as_ref().map(|(ps, _)| ps.alpha_indices.len()).unwrap_or(0),
    blocks_processed,
    total_proc_us,
);

    Ok(())
}
/// Issue user state for given addresses at snapshot block
fn cmd_issue_user_state(
    logn: usize,
    srs: PathBuf,
    snapshot: PathBuf,
    universe_file: PathBuf,
    addresses: PathBuf,
    snapshot_vals: PathBuf, 
    out: PathBuf,
) -> Result<()> {
    let t0 = t_start();
    let n = 1usize << logn;

    let lc = get_loaded_ctx(logn, &srs)?;
    let ctx = &lc.ctx;
    let srs_id = lc.srs_id.clone();

    let snap: SnapshotOut = serde_json::from_slice(&fs::read(&snapshot)?)?;
    if snap.logn != logn || snap.srs_id != srs_id {
        bail!("snapshot / srs mismatch");
    }

    let universe = read_addresses_file(&universe_file)?;
    let indexer = Indexer::from_universe_sequential(&universe, n)?;

    let addrs = read_addresses_file(&addresses)?;
    if addrs.is_empty() {
        bail!("addresses file is empty");
    }

    let mut set = std::collections::BTreeSet::new();
    for a in addrs {
        let i = indexer.index_of(a)?;
        set.insert(i);
    }
    let alpha_indices: Vec<usize> = set.into_iter().collect();



    // Load snapshot values (this is REQUIRED in Cauchy)
    let bals = snapshot_vals::read_u256hex_lines(&snapshot_vals)?;
    if bals.len() != universe.len() {
        bail!("snapshot_vals length mismatch");
    }

    let mut v = vec![Fr::ZERO; n];
    for (a, bal) in universe.iter().zip(bals.into_iter()) {
        let i = indexer.index_of(*a)?;
        v[i] = fr_from_u256_exact(bal)?;
    }

   let (gc_rebuilt, gq_alpha) =
    ctx.build_commitment_for_alpha(&v, &alpha_indices);


if g1_to_hex(&gc_rebuilt) != snap.gc_hex {
    bail!("rebuilt snapshot commitment mismatch");
}
println!("snapshot gc = {}", snap.gc_hex);
println!("rebuilt  gc = {}", g1_to_hex(&gc_rebuilt));
let alpha_len = alpha_indices.len();

   let us = UserState {
    n,
    logn,
    srs_id,
    last_block: snap.block_number,
    gc_hex: snap.gc_hex.clone(),
    alpha_indices,
    alpha_witnesses_hex: gq_alpha.iter().map(g1_to_hex).collect(),
    universe_mode: "external".into(),
    witness_mode: "from_snapshot".into(),
};

        

    fs::write(&out, serde_json::to_vec_pretty(&us)?)?;

   emit(
    "[IssueUserState]",
    snap.block_number,
    n,
    alpha_len,
    0,
    t_us(t0),
);
    Ok(())
}

/// Initialize proof server with user states
fn cmd_proof_server_init(
    logn: usize,
    srs: PathBuf,
    snapshot: PathBuf,
    user_id: Vec<String>,
    user_state: Vec<PathBuf>,
    out: PathBuf,
) -> Result<()> {
    if user_id.len() != user_state.len() {
        bail!("--user-id and --user-state length mismatch");
    }

    let lc = get_loaded_ctx(logn, &srs)?;
    let srs_id = lc.srs_id.clone();
    let snap: SnapshotOut = serde_json::from_slice(&fs::read(&snapshot)?)?;

    if snap.srs_id != srs_id || snap.logn != logn {
        bail!("snapshot / srs mismatch");
    }

    let mut users = Vec::new();
    for (uid, path) in user_id.into_iter().zip(user_state.into_iter()) {
        let u: UserState = serde_json::from_slice(&fs::read(path)?)?;

        if u.alpha_indices.len() != u.alpha_witnesses_hex.len() {
            bail!("witness length mismatch for user {}", uid);
        }

        users.push((uid, u));
    }

    let ps = ProofServerState::from_user_states(&users)?;
    if ps.gc_hex != snap.gc_hex {
        bail!("commitment mismatch with snapshot");
    }

    fs::write(&out, serde_json::to_vec_pretty(&ps)?)?;
    Ok(())
}

/// Advance proof server to new block
fn cmd_proof_server_advance(
    logn: usize,
    srs: PathBuf,
    proof_server_state: PathBuf,
    journal: PathBuf,
    end_block: u64,
    out: PathBuf,
) -> Result<()> {
    let mut ps: ProofServerState = serde_json::from_slice(&fs::read(&proof_server_state)?)?;

    let lc = get_loaded_ctx(logn, &srs)?;
    let ctx = &lc.ctx;

    let srs_id = ps.srs_id.clone();

    let mut gc: G1 = g1_from_hex(&ps.gc_hex)?;
    let mut gq: Vec<G1> = ps.alpha_witnesses_hex.iter().map(|h| g1_from_hex(h)).collect::<Result<Vec<_>>>()?;

    if ps.alpha_indices.len() != gq.len() {
        bail!("proof_server_state alpha_indices / alpha_witnesses mismatch");
    }

    let start_block = ps.last_block;
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

            if j.changed_indices.len() != j.delta_hex.len() {
                bail!("journal line malformed at block {}", j.block_number);
            }

            let mut delta = Vec::<Fr>::with_capacity(j.delta_hex.len());
            for h in &j.delta_hex {
                delta.push(fr_from_hex(h)?);
            }
            let (beta, delta) = canonicalize_beta_delta(&j.changed_indices, &delta);

            let t_commit = t_start();
            for (&i, &d) in beta.iter().zip(delta.iter()) {
                gc = ctx.update_commitment(gc, i, d);
            }
            let commit_us = t_us(t_commit);

            if let Some(pinned) = j.gc_hex.as_deref() {
                if g1_to_hex(&gc) != pinned {
                    bail!("commitment mismatch at block {}", j.block_number);
                }
            } else {
                bail!("journal line missing pinned gc_hex at block {}", j.block_number);
            }

            let t_wit = t_start();
            if !beta.is_empty() && !ps.alpha_indices.is_empty() {
                gq = ctx.update_witnesses_batch(&ps.alpha_indices, &gq, &beta, &delta);
            }
            let wit_us = t_us(t_wit);

            ps.last_block = j.block_number;

            blocks_applied += 1;
            updates_applied += beta.len();

            emit("[ProofServerAdvanceBlock]", j.block_number, ps.n, ps.alpha_indices.len(), beta.len(), t_us(t_block));
            emit("[ProofServerAdvanceCommitOnly]", j.block_number, ps.n, 0, beta.len(), commit_us);
            emit("[ProofServerAdvanceWitnessOnly]", j.block_number, ps.n, ps.alpha_indices.len(), beta.len(), wit_us);

            Ok(())
        },
    )?;

    ps.gc_hex = g1_to_hex(&gc);
    ps.alpha_witnesses_hex = gq.iter().map(g1_to_hex).collect();

    fs::write(out, serde_json::to_vec_pretty(&ps)?)?;

    let total_micros = t_us(t_total);
    emit("[ProofServerAdvanceTotal]", ps.last_block, ps.n, ps.alpha_indices.len(), updates_applied, total_micros);
    eprintln!(
        "[ProofServerAdvance] blocks_applied={} updates={} total_micros={}",
        blocks_applied, updates_applied, total_micros
    );
    Ok(())
}

fn for_each_journal_line_rev_until<F>(path: &Path, mut f: F) -> Result<()>
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

fn cancelled(cancel: Option<&std::path::Path>) -> bool {
    cancel.map(|p| p.exists()).unwrap_or(false)
}

#[derive(serde::Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum DaemonReq {
    PublisherAdvance {
        dataset_dir: PathBuf,
        universe_file: PathBuf,
        snapshot: PathBuf,
        snapshot_vals: PathBuf,
        end_block: u64,
        journal: PathBuf,
        alpha_addresses: Option<PathBuf>,
        proof_server_out: Option<PathBuf>,
        block_log_csv: Option<PathBuf>,
        cancel_path: Option<PathBuf>,
    },
    ProofServerAdvance {
        proof_server_state: PathBuf,
        journal: PathBuf,
        end_block: u64,
        out: PathBuf,
    },
    ProofServerHistory {
        proof_server_state: PathBuf,
        journal: PathBuf,
        start_block: u64,
        blocks: Vec<u64>,
        user_id: String,
        out: PathBuf,

        // optional per-request log + cancel, same as your history-server style
        log: Option<PathBuf>,
        req_id: Option<String>,
        cancel_path: Option<PathBuf>,
    },
    UserVerify {
        user_state: PathBuf,
        history: PathBuf,
        claims: Option<PathBuf>,
    },
}

#[derive(serde::Serialize)]
struct DaemonResp {
    ok: bool,
    err: Option<String>,
}

#[derive(serde::Deserialize)]
struct HistoryReq {
    start_block: u64,
    blocks: Vec<u64>,
    user_id: String,
    out: std::path::PathBuf,

    log: Option<std::path::PathBuf>,
    req_id: Option<String>,

    // ADD THIS:
    cancel_path: Option<std::path::PathBuf>,
}


fn cmd_proof_server_history(
    logn: usize,
    srs: PathBuf,
    proof_server_state: PathBuf,
    journal: PathBuf,
    start_block: u64,
    blocks: Vec<u64>,
    user_id: String,
    out: PathBuf,
) -> Result<()> {
    let t_total = t_start();

    // Load PS state
    let ps: ProofServerState = serde_json::from_slice(&fs::read(&proof_server_state)?)?;

    // Load SRS + build ctx
    let t_ctx = t_start();
    let lc = get_loaded_ctx(logn, &srs)?;
    let ctx = &lc.ctx;
    emit("[HistoryCtxLookup]", ps.last_block, ps.n, ps.alpha_indices.len(), 0, t_us(t_ctx));

    // Run core logic
    let t_core = t_start();
    proof_server_history_core(
        ctx,
        &ps,
        &journal,
        start_block,
        blocks,
        user_id,
        out,
        None,
    )?;
    emit(
        "[HistoryOneShotCore]",
        ps.last_block,
        ps.n,
        ps.alpha_indices.len(),
        0,
        t_us(t_core),
    );

    emit(
        "[HistoryOneShotTotal]",
        ps.last_block,
        ps.n,
        ps.alpha_indices.len(),
        0,
        t_us(t_total),
    );
    Ok(())
}

pub fn proof_server_history_core(
    ctx: &VcContext,
    ps: &ProofServerState,
    journal: &std::path::Path,
    start_block: u64,
    mut blocks: Vec<u64>,
    user_id: String,
    out: std::path::PathBuf,
    cancel: Option<&std::path::Path>,
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

    // ---- decode head witnesses for this user
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

    // =========================================================
    // ONE PATH ALWAYS (paper-style):
    // Build stream_rev (reverse scan + negate updates + insert queries),
    // then run VUpdate D&C, then write results.
    // Works for blocks.len()==1 and blocks.len()>1.
    // =========================================================

    let min_block = blocks[0];
    let want: HashSet<u64> = blocks.iter().copied().collect();

    let t_build = t_start();
    let mut stream_rev: Vec<HistoryOp> = Vec::new();
    let mut expected = head_block;

    let mut lines_parsed: usize = 0;
    let mut micros_json_parse: u128 = 0;
    let micros_delta_decode: u128 = 0;

    for_each_journal_line_rev_until(journal, |line| -> eyre::Result<bool> {
        if cancelled(cancel) {
            eyre::bail!("cancelled during journal scan");
        }

        // Stop once we are done down to min_block
        if expected <= min_block {
            return Ok(false);
        }

        // Parse lite JSON
        let t_j = t_start();
        let j: JournalLineLite = serde_json::from_str(line)?;
        micros_json_parse += t_us(t_j);
        lines_parsed += 1;

        // Filter irrelevant entries
        if j.block_number <= start_block || j.block_number > head_block {
            return Ok(true);
        }
        if j.srs_id != ps.srs_id || j.logn != ps.logn || j.n != ps.n {
            return Ok(true);
        }

        // GAP DETECTION after filters:
        // We must see blocks strictly descending: head, head-1, ...
        if j.block_number < expected {
            eyre::bail!(
                "journal gap: missing block {} while building history stream (next seen block {}). \
                 Journal must contain every block in the window (even if beta empty).",
                expected,
                j.block_number
            );
        }
        if j.block_number > expected {
            // unexpected extra/duplicate line; keep scanning
            return Ok(true);
        }

        // Now j.block_number == expected
        let b = expected;

        // Query at state b happens BEFORE applying Update(b) to move to b-1
        if want.contains(&b) {
            stream_rev.push(HistoryOp::Query { query_id: b as usize });
        }

        // Decode and negate deltas (rewind direction)
        let _t_d = t_start();
        let mut delta_neg = Vec::<Fr>::with_capacity(j.delta_hex.len());
        for hx in &j.delta_hex {
            delta_neg.push(-fr_from_hex(hx)?); // keep alignment 1:1 with changed_indices
        }
        let (beta, delta) = canonicalize_beta_delta(&j.changed_indices, &delta_neg);

                if !beta.is_empty() {
                    stream_rev.push(HistoryOp::Update { beta, delta });
        }

        expected -= 1;
        Ok(true)
    })?;

    // After scan, we must have reached min_block
    if expected > min_block {
        eyre::bail!(
            "journal ended early while building history stream: stopped at {}, need min_block {}. \
             Journal is incomplete for the requested window.",
            expected,
            min_block
        );
    }

    // Final query at min_block (no update after it)
    if want.contains(&min_block) {
        stream_rev.push(HistoryOp::Query { query_id: min_block as usize });
    }

    emit("[HistoryJournalJsonParseLite]", 0, n, user_alpha.len(), lines_parsed, micros_json_parse);
    emit("[HistoryDeltaDecode]", 0, n, user_alpha.len(), stream_rev.len(), micros_delta_decode);
    emit("[HistoryBuildStream]", 0, n, user_alpha.len(), stream_rev.len(), t_us(t_build));

    if cancelled(cancel) {
        eyre::bail!("cancelled before vupdate");
    }

    // ---- VUpdate (paper D&C)
    let t_vu = t_start();
    let vu = vupdate_history_vupdate_dc(ctx, &user_alpha, &user_gq_head, &stream_rev);
    emit("[HistoryVUpdateUserAlpha]", 0, n, user_alpha.len(), vu.len(), t_us(t_vu));

    // ---- payload write
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

    let gc_hex = pinned_commitment_at(
        journal,
        &ps.srs_id,
        ps.n,
        ps.logn,
        b,
    )?;

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
    emit("[HistoryPayloadWrite]", 0, n, user_alpha.len(), json.len(), t_us(t_payload));

    emit("[HistoryTotal]", 0, n, user_alpha.len(), 0, t_us(t_total));
    Ok(())
}


fn cmd_proof_server_history_server(
    logn: usize,
    srs: std::path::PathBuf,
    proof_server_state: std::path::PathBuf,
    journal: std::path::PathBuf,
    in_path: Option<std::path::PathBuf>,
) -> eyre::Result<()> {
    use std::fs::OpenOptions;
    use std::io::{self, BufRead};

    let t_srs = t_start();
    let ps: ProofServerState = serde_json::from_slice(&std::fs::read(&proof_server_state)?)?;
   let lc = get_loaded_ctx(logn, &srs)?;
    let ctx = &lc.ctx;
    
    emit("[HistoryServerCtxLookup]", ps.last_block, ps.n, ps.alpha_indices.len(), 0, t_us(t_srs));

    eprintln!("history-server ready: head_block={}", ps.last_block);

    // IMPORTANT: if input is a FIFO, open it read+write so it never sees EOF
    // just because the writer closes.
    let reader: Box<dyn BufRead> = if let Some(p) = in_path {
        let f = OpenOptions::new()
            .read(true)
            .write(true) // key for FIFO: prevents EOF shutdown
            .open(p)?;
        Box::new(io::BufReader::new(f))
    } else {
        Box::new(io::BufReader::new(io::stdin()))
    };

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }

        let req: HistoryReq = serde_json::from_str(&line)?;

        // open per-request log if provided
        let mut lf = match &req.log {
            Some(p) => Some(std::fs::File::create(p)?),
            None => None,
        };

        // Write a small header into the per-request log
        if let Some(w) = lf.as_mut() {
            let rid = req.req_id.clone().unwrap_or_else(|| "-".to_string());
            log_line(w, &format!("REQ id={} user={} out={}", rid, req.user_id, req.out.display()));
            log_line(w, &format!("REQ start_block={} blocks={:?}", req.start_block, req.blocks));
            emit_to(w, "[HistoryReqBegin]", ps.last_block, ps.n, 0, req.blocks.len(), 0);
        }

        let t_req = t_start();
        let r = proof_server_history_core_with_log(
            ctx,
            &ps,
            &journal,
            req.start_block,
            req.blocks,
            req.user_id,
            req.out,
            lf.as_mut(),
            req.cancel_path.as_deref(), // <-- ADD
        );
        

        if let Some(w) = lf.as_mut() {
            emit_to(w, "[HistoryReqEnd]", ps.last_block, ps.n, 0, 0, t_us(t_req));
        }

        match r {
            Ok(()) => {
                eprintln!("ok");
            }
            Err(e) => {
                // log error in server stderr and (if available) per-request log
                eprintln!("err: {:?}", e);
                // (lf already dropped here; if you want, you can keep it and write error too)
            }
        }
    }

    Ok(())
}


fn pinned_commitment_at(journal: &Path, srs_id: &str, n: usize, logn: usize, block: u64) -> Result<String> {
    let mut out: Option<String> = None;
    buvc_rs::journal::for_each_line_filtered(journal, srs_id, n, logn, |j: JournalLine| {
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


/// Measure SINGLE proof verification (α = 1)
fn bench_verify_single_only(
    ctx: &VcContext,
    vp: &VcParameter,
    gc: G1,
    index: usize,
    value: Fr,
    gq: G1,
) -> bool {
    let t = Instant::now();
    let ok = ctx.verify(vp, gc, index, value, gq);
    let us = t.elapsed().as_micros();

    println!(
        "[VERIFY-SINGLE] alpha=1 index={} verify_us={} ok={}",
        index, us, ok
    );
    ok
}

/// Measure MULTI proof verification (α ≥ 1)
fn bench_verify_multi_only(
    ctx: &VcContext,
    vp: &VcParameter,
    gc: G1,
    indices: &[usize],
    values: &[Fr],
    witnesses: &[G1],
) -> bool {
    // aggregation (paper counts this)
    let t_agg = Instant::now();
    let agg = ctx.aggregate_proof(indices, witnesses);
    let agg_us = t_agg.elapsed().as_micros();

    // verification
    let t_ver = Instant::now();
    let ok = ctx.verify_multi(vp, gc, indices, values, agg);
    let ver_us = t_ver.elapsed().as_micros();

    println!(
        "[VERIFY-MULTI] alpha={} aggregate_us={} verify_us={} total_us={} ok={}",
        indices.len(),
        agg_us,
        ver_us,
        agg_us + ver_us,
        ok
    );
    ok
}


/// Verify user state against history and claims
fn cmd_user_verify(
    logn: usize,
    srs: PathBuf,
    user_state: PathBuf,
    history: PathBuf,
    claims: Option<PathBuf>,
) -> Result<()> {
    let hist: Vec<ProofServerHistoryOut> = serde_json::from_slice(&fs::read(&history)?)?;
    if hist.is_empty() {
        bail!("empty history");
    }

    let us: UserState = serde_json::from_slice(&fs::read(&user_state)?)?;
    if us.logn != logn {
        bail!("user_state logn mismatch");
    }

    let t_lookup = t_start();
    let lc = get_loaded_ctx(logn, &srs)?;
    let ctx = &lc.ctx;
    let vp  = &lc.vp;
    let srs_id = &lc.srs_id;
    let lookup_us = t_us(t_lookup);
    
    // Optional: keep a metric, but it's now "lookup", not "load"
    emit("[UserVerifyCtxLookup]", 0, 0, 0, 0, lookup_us);

    if us.srs_id != *srs_id {
        bail!("SRS id mismatch user_state ({}) vs srs.bin ({})", us.srs_id, srs_id);
    }
    let _n = 1usize << logn;
    let mut claim_map: HashMap<u64, Vec<Fr>> = HashMap::new();
    if let Some(p) = claims {
        let v: Vec<ValueClaim> = serde_json::from_slice(&fs::read(&p)?)?;
        for c in v {
            let mut vals = Vec::<Fr>::with_capacity(c.values_hex.len());
            for hx in c.values_hex {
                vals.push(fr_from_hex(&hx)?);
            }
            claim_map.insert(c.block, vals);
        }
    } else {
        bail!(
            "No values available for verification.\n\
             Provide --claims <json> (recommended)."
        );
    }

    let _total_alpha = 0usize;
    let _proofs = 0usize;
    let _t_total = t_start();
for h in hist {
    if h.indices != us.alpha_indices {
        bail!("history indices != user_state alpha_indices at block {}", h.block);
    }
    if h.witnesses_hex.len() != h.indices.len() {
        bail!("witness length mismatch at block {}", h.block);
    }

    // ---- public commitment (manually pinned)
    let gc_hex = match h.block {
        18908896 => "0x8db513e2799b92592856a9324793e744b9ed745c6e79178e6cbb25bc2e1f0ff7cf3cee8342ec7c575d239bd9c96fbef4",
        18908897 => "0x83e994ce642454c80e4f65480e6643f5820c48cc0a11d382abe97ad949f1ba4d2b21bf1bf7ba4c082d261a6463686913",
        18912000 => "0xb71ab951fdb1ceaedece9955e90936af13fa83c0303ac78f33f86216e6f894e09d4e3fe0574c833b36abe0890e5b7e2b",
        18912001 => "0x975c77b723c5fbcae9b17fcf9c0a16d63007bca8d9751e2e563c7571184c5f13e67f1db7526e65f5ee5715bf8d116321",
        18915995 => "0x960710a153b0f5717b8bcbbe1d581d9ad6a162064702cd1e55503b4c8b95a7f7aea7f0f216d07c718f4e3995fb0137aa",
        18915996 => "0x9469859c35ad9089bb7943a210177cbb44714fb59f88ba25fc7b03b21475a20d1403fbdef8a9fd7a2fd5b6c223495284",
        18915997 => "0xb0cb7b6a44947be11cb377090843ac516269ca0e2259c28acf626307788d82f772142697397f1bf7de1c8d0f0b6df65c",
        18915998 => "0x8e8c629607cd612eb7dd5e104f03f6e7c05ed46aa39dc5d337378c129ed141fea70bc4aad3414d716ea2e33a505ba234",
        18915999 => "0x8340938cc0ccea605fbfd19b8b89b4c104ab16eae6962060238d199172f4bc5979f5ad44507d61a72430517a824c9591",
        18916000 => "0xb9a111f6742eb26a81eb03b97bd148c2c67b4e401b9653331e867f49d5a45a64ad455469c2f88c9cff586c216bbd19e1",
        _ => bail!("missing gc_hex for block {}", h.block),
    };
    let gc = g1_from_hex(gc_hex)?;

    // ---- claimed values
    let vals = claim_map
        .get(&h.block)
        .ok_or_else(|| eyre::eyre!("missing value claim for block {}", h.block))?;

    if vals.len() != h.indices.len() {
        bail!("value claim length mismatch at block {}", h.block);
    }

    // ---- decode witnesses ONCE
    let mut gqs = Vec::<G1>::with_capacity(h.witnesses_hex.len());
    for w_hex in &h.witnesses_hex {
        gqs.push(g1_from_hex(w_hex)?);
    }

    // =====================================================
    // 🔥 PAPER-FAITHFUL MEASUREMENTS
    // =====================================================

    // α = 1 → single-proof verification
    if h.indices.len() == 1 {
        bench_verify_single_only(
            ctx,
            vp,
            gc,
            h.indices[0],
            vals[0],
            gqs[0],
        );
    }

    // α ≥ 1 → multi-proof (aggregate + verify)
    bench_verify_multi_only(
        ctx,
        vp,
        gc,
        &h.indices,
        vals,
        &gqs,
    );
}
    Ok(())
}

fn proof_server_history_core_with_log(
    ctx: &VcContext,
    ps: &ProofServerState,
    journal: &std::path::Path,
    start_block: u64,
    blocks: Vec<u64>,
    user_id: String,
    out: std::path::PathBuf,
    mut log: Option<&mut std::fs::File>,
    cancel: Option<&std::path::Path>, // <-- ADD
) -> eyre::Result<()> {
    let t0 = t_start();

    if cancelled(cancel) {
        eyre::bail!("cancelled before start");
    }

    if let Some(w) = log.as_mut() {
        emit_to(w, "[HistoryCoreEnter]", ps.last_block, ps.n, 0, blocks.len(), 0);
    }

    let r = proof_server_history_core(ctx, ps, journal, start_block, blocks, user_id, out, cancel);

    if let Some(w) = log.as_mut() {
        emit_to(w, "[HistoryCoreExit]", ps.last_block, ps.n, 0, 0, t_us(t0));
    }

    r
}
fn cmd_extract_value(
    dataset_dir: PathBuf,
    snapshot: PathBuf,
    snapshot_vals: PathBuf,
    universe_file: PathBuf,
    target_block: u64,
    index: usize,
) -> Result<()> {
    let snap: SnapshotOut = serde_json::from_slice(&fs::read(&snapshot)?)?;

    if target_block < snap.block_number {
        bail!(
            "target_block {} < snapshot block {}",
            target_block,
            snap.block_number
        );
    }

    // Load universe
    let universe = read_addresses_file(&universe_file)?;
    let addr = universe
        .get(index)
        .ok_or_else(|| eyre::eyre!("index {} out of universe bounds", index))?
        .clone();

    // Replay state
    let mut tracker = StateTracker::from_snapshot_vals_file(
        &dataset_dir,
        100_000,
        snap.block_number,
        &universe,
        &snapshot_vals,
    )?;

    while tracker.cur_block < target_block {
        tracker.apply_next_block()?;
    }

    // 🔑 Correct: lookup by address, already Fr
    let val_fr = tracker
        .balances
        .get(&addr)
        .ok_or_else(|| eyre::eyre!("missing balance for address {:?}", addr))?;

    println!("{}", fr_to_hex(val_fr));
    Ok(())
}

fn cmd_daemon(logn: usize, srs: PathBuf, in_path: Option<PathBuf>) -> Result<()> {
    use std::fs::OpenOptions;
    use std::io::{self, BufRead};

    // Force single init now (so first request isn't charged)
    let _ = get_loaded_ctx(logn, &srs)?;

    let reader: Box<dyn BufRead> = if let Some(p) = in_path {
        let f = OpenOptions::new().read(true).write(true).open(p)?;
        Box::new(io::BufReader::new(f))
    } else {
        Box::new(io::BufReader::new(io::stdin()))
    };

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() { continue; }

        let req: DaemonReq = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                println!("{}", serde_json::to_string(&DaemonResp{ ok:false, err:Some(e.to_string()) })?);
                continue;
            }
        };

        let r: Result<()> = match req {
            DaemonReq::PublisherAdvance {
                dataset_dir,
                universe_file,
                snapshot,
                snapshot_vals,
                end_block,
                journal,
                alpha_addresses,
                proof_server_out,
                block_log_csv,
                cancel_path,
            } => cmd_publisher_advance(
                logn,
                srs.clone(),
                dataset_dir,
                universe_file,
                snapshot,
                snapshot_vals,
                end_block,
                journal,
                alpha_addresses,
                proof_server_out,
                block_log_csv,
                cancel_path,
            ),
          
            DaemonReq::ProofServerAdvance { proof_server_state, journal, end_block, out } =>
                cmd_proof_server_advance(logn, srs.clone(), proof_server_state, journal, end_block, out),
          
            DaemonReq::ProofServerHistory { proof_server_state, journal, start_block, blocks, user_id, out, .. } =>
                cmd_proof_server_history(logn, srs.clone(), proof_server_state, journal, start_block, blocks, user_id, out),
          
            DaemonReq::UserVerify { user_state, history, claims } =>
                cmd_user_verify(logn, srs.clone(), user_state, history, claims),
          };

        match r {
            Ok(()) => println!("{}", serde_json::to_string(&DaemonResp{ ok:true, err:None })?),
            Err(e) => println!("{}", serde_json::to_string(&DaemonResp{ ok:false, err:Some(format!("{:?}", e)) })?),
        }
    }

    Ok(())
}