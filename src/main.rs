// src/main.rs
use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::{Field, Zero};
use clap::{Parser, Subcommand};
use ethers_core::types::Address;
use eyre::{bail, Result};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{BufRead, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use buvc_rs::codec::{
    delta_fr, fr_from_hex, fr_from_u256_exact, fr_to_hex, g1_from_hex, g1_to_hex
};
use buvc_rs::snapshot_vals;
use buvc_rs::StateTracker;
use buvc_rs::history::vupdate_history_same_alpha;
use buvc_rs::HistoryOp;
use buvc_rs::Indexer;
use buvc_rs::journal::append_line;
use buvc_rs::srs::{load_or_create_srs, make_ctx};
use buvc_rs::{JournalLine, SnapshotOut, UserState};
use buvc_rs::ProofServerState;


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

#[derive(Clone, Debug, serde::Deserialize)]
struct JournalLineLite {
    pub block_number: u64,
    pub n: usize,
    pub logn: usize,
    pub srs_id: String,
    pub changed_indices: Vec<usize>,
    pub delta_hex: Vec<String>,
}

fn emit(phase: &'static str, block: u64, n: usize, alpha: usize, beta: usize, micros: u128) {
    let m = Metric { phase, block, n, alpha, beta, micros };
    println!("METRIC {}", serde_json::to_string(&m).unwrap());
}

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
        addresses: PathBuf,
        #[arg(long)]
        snapshot_vals: PathBuf, 
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
        user_id: Vec<String>,
        #[arg(long)]
        user_state: Vec<PathBuf>,
    
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

    ProofServerExportUserState {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        proof_server_state: PathBuf,
        #[arg(long)]
        user_id: String,
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
        journal: PathBuf,
        #[arg(long)]
        user_state: PathBuf,
        #[arg(long)]
        history: PathBuf,
        #[arg(long)]
        claims: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    color_eyre::install().ok();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::BuildUniverse { dataset_dir, start_block, end_block, out } => {
            cmd_build_universe(dataset_dir, start_block, end_block, out)
        }
    
        Cmd::PublisherSnapshot { logn, srs, block, universe_file, snapshot_vals, out } => {
            cmd_publisher_snapshot(logn, srs, block, universe_file, snapshot_vals, out)
        }
    
        Cmd::PublisherAdvance { logn, srs, dataset_dir, universe_file, snapshot, snapshot_vals, end_block, journal } => {
            cmd_publisher_advance(logn, srs, dataset_dir, universe_file, snapshot, snapshot_vals, end_block, journal)
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

            )
        }
        Cmd::ProofServerInit { logn, srs, snapshot, user_id, user_state, out } => {
            cmd_proof_server_init(logn, srs, snapshot, user_id, user_state, out)
        }
    
        Cmd::ProofServerAdvance { logn, srs, proof_server_state, journal, end_block, out } => {
            cmd_proof_server_advance(logn, srs, proof_server_state, journal, end_block, out)
        }
    
        Cmd::ProofServerExportUserState { logn, srs, proof_server_state, user_id, out } => {
            cmd_proof_server_export_user_state(logn, srs, proof_server_state, user_id, out)
        }
    
        Cmd::ProofServerHistory { logn, srs, proof_server_state, journal, start_block, blocks, user_id, out } => {
            cmd_proof_server_history(logn, srs, proof_server_state, journal, start_block, blocks, user_id, out)
        }
    
        Cmd::UserVerify { logn, srs, journal, user_state, history, claims } => {
            cmd_user_verify(logn, srs, journal, user_state, history, claims)
        }
    }
}

fn cmd_build_universe(dataset_dir: PathBuf, start_block: u64, end_block: u64, out: PathBuf) -> Result<()> {
    let mut dataset = buvc_rs::DatasetReader::new(&dataset_dir, 100_000);
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

fn cmd_publisher_snapshot(
    logn: usize,
    srs: PathBuf,
    block: u64,
    universe_file: PathBuf,
    snapshot_vals_path: PathBuf,
    out: PathBuf,
) -> Result<()> {
    let t0 = t_start();

    let n = 1usize << logn;
    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    let ctx = make_ctx(&vp, logn);

    let universe = read_addresses_file(&universe_file)?;
    let bals = snapshot_vals::read_u256hex_lines(&snapshot_vals_path)?;
    if universe.len() != bals.len() {
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

    let micros = t_us(t0);
    let sz = fs::metadata(&out)?.len() as usize;
    emit("[PublisherSnapshot]", block, n, 0, sz, micros);
    Ok(())
}

fn cmd_publisher_advance(
    logn: usize,
    srs: PathBuf,
    dataset_dir: PathBuf,
    universe_file: PathBuf,
    snapshot: PathBuf,
    snapshot_vals_path: PathBuf,
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
        &snapshot_vals_path,
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
                gc_hex: Some(g1_to_hex(&gc)), // publisher pinned commitment
            },
        )?;

        emit("[PublisherAdvance] block", cur, n, 0, delta.len(), t_us(t0));
        emit("[CommitUpdate] block", cur, n, 0, delta.len(), t2);
    }
    
    Ok(())
}

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

    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    let ctx = make_ctx(&vp, logn);

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

    // Paper-faithful: compute commitment + witness together
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

    let (_vp, srs_id) = load_or_create_srs(&srs, logn)?;
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

fn cmd_proof_server_advance(
    logn: usize,
    srs: PathBuf,
    proof_server_state: PathBuf,
    journal: PathBuf,
    end_block: u64,
    out: PathBuf,
) -> Result<()> {
    let mut ps: ProofServerState = serde_json::from_slice(&fs::read(&proof_server_state)?)?;

    let (vp, _) = load_or_create_srs(&srs, logn)?;
    let ctx = make_ctx(&vp, logn);

    let mut gc: G1 = g1_from_hex(&ps.gc_hex)?;
    let mut gq: Vec<G1> = ps.alpha_witnesses_hex.iter().map(|h| g1_from_hex(h)).collect::<Result<Vec<_>>>()?;

    if ps.alpha_indices.len() != gq.len() {
        bail!("proof_server_state alpha_indices / alpha_witnesses mismatch");
    }

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

fn cmd_proof_server_export_user_state(
    logn: usize,
    srs: PathBuf,
    proof_server_state: PathBuf,
    user_id: String,
    out: PathBuf,
) -> Result<()> {
    let t0 = t_start();

    let ps: ProofServerState = serde_json::from_slice(&fs::read(&proof_server_state)?)?;
    if ps.logn != logn {
        bail!("logn mismatch");
    }
    if ps.alpha_indices.len() != ps.alpha_witnesses_hex.len() {
        bail!("alpha_indices / alpha_witnesses mismatch");
    }

    let (_vp, srs_id) = load_or_create_srs(&srs, logn)?;
    if ps.srs_id != srs_id {
        bail!("srs_id mismatch");
    }

    let (user_alpha, user_pos_in_union) = ps
        .user_alpha(&user_id)
        .ok_or_else(|| eyre::eyre!("unknown user_id"))?;

    let mut gq_head: Vec<G1> = Vec::with_capacity(user_pos_in_union.len());
    for &p in &user_pos_in_union {
        gq_head.push(g1_from_hex(&ps.alpha_witnesses_hex[p])?);
    }
    let us = UserState {
        n: ps.n,
        logn: ps.logn,
        srs_id: ps.srs_id.clone(),
    
        last_block: ps.last_block,

        gc_hex: String::new(),
    
        alpha_indices: user_alpha.clone(),
    
        alpha_witnesses_hex: gq_head.iter().map(g1_to_hex).collect(),
    
        universe_mode: "external".to_string(),
        witness_mode: "export_from_proof_server".to_string(),
    };
    
    
    fs::write(&out, serde_json::to_vec_pretty(&us)?)?;
    let micros = t_us(t0);
    let sz = fs::metadata(&out)?.len() as usize;
    emit("[ProofServerExportUserState]", ps.last_block, ps.n, user_alpha.len(), sz, micros);
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ProofServerHistoryOut {
    pub user_id: String,
    pub block: u64,
    pub indices: Vec<usize>,
    pub witnesses_hex: Vec<String>,
}

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

    let ps: ProofServerState = serde_json::from_slice(&fs::read(&proof_server_state)?)?;
    if ps.logn != logn {
        bail!("logn mismatch");
    }
    if ps.alpha_indices.len() != ps.alpha_witnesses_hex.len() {
        bail!("alpha_indices / alpha_witnesses mismatch");
    }

    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    if ps.srs_id != srs_id {
        bail!("srs_id mismatch");
    }
    let ctx = make_ctx(&vp, logn);

    let (user_alpha, user_pos_in_union) = ps
        .user_alpha(&user_id)
        .ok_or_else(|| eyre::eyre!("unknown user_id"))?;

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

    let head_block = ps.last_block;
    let min_block = blocks[0];
    let want: HashSet<u64> = blocks.iter().copied().collect();

    let t_decode = t_start();
    let mut user_gq_head: Vec<G1> = Vec::with_capacity(user_pos_in_union.len());
    for &p in &user_pos_in_union {
        user_gq_head.push(g1_from_hex(&ps.alpha_witnesses_hex[p])?);
    }
    emit("[HeadUserDecode]", head_block, ps.n, user_alpha.len(), user_alpha.len(), t_us(t_decode));

    let t_build = t_start();
    let mut stream_rev: Vec<HistoryOp> = Vec::new();
    let mut cur = head_block;

    let mut lines_parsed: usize = 0;
    let mut micros_json_parse: u128 = 0;
    let mut micros_delta_decode: u128 = 0;

    for_each_journal_line_rev_until(&journal, |line| -> Result<bool> {
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

        while cur > j.block_number {
            if want.contains(&cur) {
                stream_rev.push(HistoryOp::Query { query_id: cur as usize });
            }
            if cur == min_block {
                return Ok(false);
            }
            cur -= 1;
        }

        if want.contains(&cur) {
            stream_rev.push(HistoryOp::Query { query_id: cur as usize });
        }
        if cur == min_block {
            return Ok(false);
        }

        let t_d = t_start();
        let mut delta = Vec::<Fr>::with_capacity(j.delta_hex.len());
        for h in &j.delta_hex {
            let mut d = fr_from_hex(h)?;
            d = -d; // rewind
            delta.push(d);
        }
        micros_delta_decode += t_us(t_d);

        let (beta, delta) = canonicalize_beta_delta(&j.changed_indices, &delta);
        if !beta.is_empty() {
            stream_rev.push(HistoryOp::Update { beta, delta });
        }

        cur -= 1;
        Ok(cur >= min_block)
    })?;

    while cur >= min_block {
        if want.contains(&cur) {
            stream_rev.push(HistoryOp::Query { query_id: cur as usize });
        }
        if cur == min_block {
            break;
        }
        cur -= 1;
    }

    emit("[HistoryJournalJsonParseLite]", 0, ps.n, user_alpha.len(), lines_parsed, micros_json_parse);
    emit("[HistoryDeltaDecode]", 0, ps.n, user_alpha.len(), stream_rev.len(), micros_delta_decode);
    emit("[HistoryBuildStream]", 0, ps.n, user_alpha.len(), stream_rev.len(), t_us(t_build));

    let t_vu = t_start();
    let vu = vupdate_history_same_alpha(&ctx, &user_alpha, &user_gq_head, &stream_rev);
    emit("[HistoryVUpdateUserAlpha]", 0, ps.n, user_alpha.len(), vu.len(), t_us(t_vu));

    let mut by_block: HashMap<u64, Vec<G1>> = HashMap::new();
    for r in vu {
        by_block.insert(r.query_id as u64, r.proofs);
    }

    let t_payload = t_start();
    let mut out_all = Vec::<ProofServerHistoryOut>::new();
    for &b in &blocks {
        let wits = by_block.get(&b).ok_or_else(|| eyre::eyre!("missing witnesses for block {}", b))?;
        if wits.len() != user_alpha.len() {
            bail!("witness length mismatch at block {}", b);
        }
        out_all.push(ProofServerHistoryOut {
            user_id: user_id.clone(),
            block: b,
            indices: user_alpha.clone(),
            witnesses_hex: wits.iter().map(g1_to_hex).collect(),
        });
    }
    out_all.sort_by_key(|o| o.block);
    let json = serde_json::to_vec_pretty(&out_all)?;
    fs::write(&out, &json)?;

    emit("[HistoryPayloadWrite]", 0, ps.n, user_alpha.len(), json.len(), t_us(t_payload));
    emit("[HistoryTotal]", 0, ps.n, user_alpha.len(), 0, t_us(t_total));
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

/* ------------------------------------------------------------- */
/* External value claims for verification                          */
/* ------------------------------------------------------------- */

#[derive(Clone, Debug, serde::Deserialize)]
struct ValueClaim {
    pub block: u64,
    pub values_hex: Vec<String>,
}

fn cmd_user_verify(
    logn: usize,
    srs: PathBuf,
    journal: PathBuf,
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

    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    if us.srs_id != srs_id {
        bail!("SRS id mismatch user_state ({}) vs srs.bin ({})", us.srs_id, srs_id);
    }
    let ctx = make_ctx(&vp, logn);
    let n = 1usize << logn;
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

    let mut total_alpha = 0usize;
    let mut proofs = 0usize;
    let t_total = t_start();

    for h in hist {
        if h.indices != us.alpha_indices {
            bail!("history indices != user_state alpha_indices at block {}", h.block);
        }
        if h.witnesses_hex.len() != h.indices.len() {
            bail!("witness length mismatch at block {}", h.block);
        }

        let gc_hex = pinned_commitment_at(&journal, &us.srs_id, n, logn, h.block)?;
        let gc = g1_from_hex(&gc_hex)?;

        let vals = claim_map.get(&h.block).ok_or_else(|| eyre::eyre!("missing value claim for block {}", h.block))?;
        if vals.len() != h.indices.len() {
            bail!("value claim length mismatch at block {}", h.block);
        }

        let mut gqs = Vec::<G1>::with_capacity(h.witnesses_hex.len());
        for w_hex in &h.witnesses_hex {
            gqs.push(g1_from_hex(w_hex)?);
        }

        let t0 = t_start();
        let agg = ctx.aggregate_proof(&h.indices, &gqs);
        let ok = ctx.verify_multi(&vp, gc, &h.indices, vals, agg);
        let dt = t_us(t0);

        emit("[UserVerify]", h.block, n, h.indices.len(), 0, dt);
        println!("user={} block={} verify_multi={} micros={}", h.user_id, h.block, ok, dt);

        total_alpha += h.indices.len();
        proofs += 1;
    }

    emit("[UserVerifyTotal]", 0, n, total_alpha, proofs, t_us(t_total));
    Ok(())
}
