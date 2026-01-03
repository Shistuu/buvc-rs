// src/main.rs
use buvc_rs::codec::{
    delta_fr, fr_from_hex, fr_from_u256_exact, fr_to_hex, g1_from_hex, g1_to_hex, parse_addr_list,
};
use buvc_rs::dataset_helpers::StateTracker;
use buvc_rs::history::{vupdate_history, HistoryOp};
use buvc_rs::indexer::Indexer;
use buvc_rs::journal::{append_line, iter_lines_filtered, map_by_block, pinned_gc_at};
use buvc_rs::srs::{load_or_create_srs, make_ctx};
use buvc_rs::types::{JournalLine, QueryAtOut, SingleProof, SnapshotOut, SnapshotState, UserState};
use buvc_rs::vc_context::VcContext;

use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::{Field, Zero};
use clap::{Parser, Subcommand};
use ethers_core::types::{Address, U256};
use eyre::{bail, Result};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

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
fn emit(phase: &'static str, block: u64, n: usize, alpha: usize, beta: usize, micros: u128) {
    let m = Metric {
        phase,
        block,
        n,
        alpha,
        beta,
        micros,
    };
    println!("METRIC {}", serde_json::to_string(&m).unwrap());
}

/// Combine duplicate indices by summing deltas. Keeps ascending order.
fn canonicalize_beta_delta(beta: &[usize], delta: &[Fr]) -> (Vec<usize>, Vec<Fr>) {
    let mut acc: BTreeMap<usize, Fr> = BTreeMap::new();
    for (&i, &d) in beta.iter().zip(delta.iter()) {
        if d.is_zero() {
            continue;
        }
        *acc.entry(i).or_insert(Fr::ZERO) += d;
    }
    acc.into_iter().unzip()
}

/// Paper-faithful step:
/// Publisher updates commitment C with (β,Δ).
/// User updates witnesses for their α with VUpdate (batch).
fn vupdate_step(
    ctx: &VcContext,
    mut gc: G1,
    alpha: &[usize],
    mut gq_alpha: Vec<G1>,
    beta: &[usize],
    delta: &[Fr],
    metric_block: u64,
    n: usize,
) -> (G1, Vec<G1>) {
    if beta.is_empty() {
        return (gc, gq_alpha);
    }

    eprintln!("[VUpdate] Block {}: Starting commitment update for {} changed indices (beta), {} alpha witnesses", metric_block, beta.len(), alpha.len());
    let t_gc = t_start();
    for (&i, &d) in beta.iter().zip(delta.iter()) {
        if !d.is_zero() {
            gc = ctx.update_commitment(gc, i, d);
        }
    }
    let commit_time = t_us(t_gc);
    emit(
        "CommitUpdate",
        metric_block,
        n,
        alpha.len(),
        beta.len(),
        commit_time,
    );
    eprintln!("[VUpdate] Block {}: Commitment update completed ({} us)", metric_block, commit_time);

    if !alpha.is_empty() {
        eprintln!("[VUpdate] Block {}: Starting witness batch update for {} alpha indices", metric_block, alpha.len());
        let t_w = t_start();
        gq_alpha = ctx.update_witnesses_batch(alpha, &gq_alpha, beta, delta);
        let witness_time = t_us(t_w);
        emit(
            "VUpdate",
            metric_block,
            n,
            alpha.len(),
            beta.len(),
            witness_time,
        );
        eprintln!("[VUpdate] Block {}: Witness batch update completed ({} us)", metric_block, witness_time);
    } else {
        eprintln!("[VUpdate] Block {}: No alpha indices to update, skipping witness update", metric_block);
    }

    (gc, gq_alpha)
}

fn read_addresses_file(path: &Path) -> eyre::Result<Vec<Address>> {
    let f = std::fs::File::open(path)?;
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

/// Ensure α ⊆ universe(indexer); return α sorted by index.
fn build_alpha_strict(
    alpha_addrs: &[Address],
    indexer: &Indexer,
) -> Result<(Vec<usize>, Vec<String>, Vec<Address>)> {
    if alpha_addrs.is_empty() {
        bail!("|α| must be > 0");
    }

    let mut tmp: Vec<(usize, Address)> = Vec::with_capacity(alpha_addrs.len());
    for &a in alpha_addrs {
        let idx = indexer.index_of(a)?;
        tmp.push((idx, a));
    }

    tmp.sort_by_key(|(i, _)| *i);

    for w in tmp.windows(2) {
        if w[0].0 == w[1].0 {
            bail!("duplicate α index {}", w[0].0);
        }
    }

    let indices = tmp.iter().map(|(i, _)| *i).collect::<Vec<_>>();
    let addrs_sorted = tmp.iter().map(|(_, a)| *a).collect::<Vec<_>>();
    let hex = addrs_sorted
        .iter()
        .map(|a| format!("{:#x}", a))
        .collect::<Vec<_>>();

    Ok((indices, hex, addrs_sorted))
}

/// Apply (β,Δ) to α-values using only deltas (user maintains values forward).
fn apply_to_alpha_values(
    alpha_pos: &HashMap<usize, usize>,
    alpha_vals: &mut [Fr],
    beta: &[usize],
    delta: &[Fr],
) {
    for (&b, &d) in beta.iter().zip(delta.iter()) {
        if let Some(&p) = alpha_pos.get(&b) {
            alpha_vals[p] += d;
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "cauchy-runner",
    about = "Paper-faithful CauchyProofs runner: publisher publishes commitments+updates; users maintain witnesses for their α via VUpdate"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Publisher: create a deterministic universe file from dataset blocks
    BuildUniverse {
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        dataset_segment_size: u32,
        #[arg(long)]
        start_block: u64,
        #[arg(long)]
        end_block: u64,
        #[arg(long, default_value = "universe.txt")]
        out: PathBuf,
    },

    /// Publisher: export snapshot_state.json (application data) at snapshot block
    SnapshotExport {
        #[arg(long)]
        snapshot_state: PathBuf,
    },
    /// Debug: dump decoded entries for a single block from dataset_dir
    DumpBlock {
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        dataset_segment_size: u32,
        #[arg(long)]
        block: u64,
    },
    /// Publisher: create snapshot.json containing ONLY the commitment C_{B0}
    PublisherSnapshot {
        #[arg(long)]
        logn: usize,
        #[arg(long, default_value = "srs.bin")]
        srs: PathBuf,
        #[arg(long)]
        block: u64,
        #[arg(long)]
        universe_file: PathBuf,
        #[arg(long)]
        snapshot_state: PathBuf,
        #[arg(long, default_value = "snapshot.json")]
        out: PathBuf,
    },

    /// Publisher: publish updates forward from snapshot block to end_block (journal.jsonl)
    PublisherAdvance {
        #[arg(long)]
        logn: usize,
        #[arg(long, default_value = "srs.bin")]
        srs: PathBuf,
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        dataset_segment_size: u32,
        #[arg(long)]
        universe_file: PathBuf,
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        snapshot_state: PathBuf,
        #[arg(long)]
        end_block: u64,
        #[arg(long, default_value = "journal.jsonl")]
        journal: PathBuf,
    },

    /// Publisher: issue initial user state for a user's α at snapshot block.
    /// This is where the publisher provides values and witnesses for α (paper-faithful).
    IssueUserState {
        #[arg(long)]
        logn: usize,
        #[arg(long, default_value = "srs.bin")]
        srs: PathBuf,
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        universe_file: PathBuf,
        #[arg(long)]
        snapshot_state: PathBuf,

        #[arg(long)]
        alpha_addresses: Option<String>,
        #[arg(long)]
        alpha_file: Option<PathBuf>,

        #[arg(long, default_value = "user_state.json")]
        out: PathBuf,
    },

    /// User: advance maintained state to requested blocks and output proofs (forward-only)
    UserQuery {
        #[arg(long)]
        logn: usize,
        #[arg(long, default_value = "srs.bin")]
        srs: PathBuf,

        #[arg(long)]
        user_state: PathBuf,
        #[arg(long, default_value = "journal.jsonl")]
        journal: PathBuf,

        #[arg(long, value_delimiter = ',')]
        blocks: Vec<u64>,

        #[arg(long, default_value = "query_at.json")]
        out: PathBuf,
    },

    /// User: compute historical witnesses for α at given blocks using
    /// the stream-based VUpdate(S, α) history algorithm.
    ///
    /// Precondition:
    ///   - user_state.last_block is the final block of the history window.
    ///   - `start_block` is the beginning of the history window (strictly less than last_block).
    ///   - `blocks` are query points with start_block < block ≤ user_state.last_block.
    UserHistory {
        #[arg(long)]
        logn: usize,
        #[arg(long, default_value = "srs.bin")]
        srs: PathBuf,

        #[arg(long)]
        user_state: PathBuf,
        #[arg(long, default_value = "journal.jsonl")]
        journal: PathBuf,

        #[arg(long)]
        start_block: u64,

        #[arg(long, value_delimiter = ',')]
        blocks: Vec<u64>,

        #[arg(long, default_value = "history_proofs.json")]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    color_eyre::install().ok();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::BuildUniverse {
            dataset_dir,
            dataset_segment_size,
            start_block,
            end_block,
            out,
        } => cmd_build_universe(
            dataset_dir,
            dataset_segment_size,
            start_block,
            end_block,
            out,
        ),

        Cmd::SnapshotExport { .. } => {
            bail!("SnapshotExport is deprecated in this paper-faithful flow. Use your existing snapshot_state.json generator.");
        }

        Cmd::PublisherSnapshot {
            logn,
            srs,
            block,
            universe_file,
            snapshot_state,
            out,
        } => cmd_publisher_snapshot(logn, srs, block, universe_file, snapshot_state, out),

        Cmd::PublisherAdvance {
            logn,
            srs,
            dataset_dir,
            dataset_segment_size,
            universe_file,
            snapshot,
            snapshot_state,
            end_block,
            journal,
        } => cmd_publisher_advance(
            logn,
            srs,
            dataset_dir,
            dataset_segment_size,
            universe_file,
            snapshot,
            snapshot_state,
            end_block,
            journal,
        ),

        Cmd::IssueUserState {
            logn,
            srs,
            snapshot,
            universe_file,
            snapshot_state,
            alpha_addresses,
            alpha_file,
            out,
        } => cmd_issue_user_state(
            logn,
            srs,
            snapshot,
            universe_file,
            snapshot_state,
            alpha_addresses,
            alpha_file,
            out,
        ),

        Cmd::UserQuery {
            logn,
            srs,
            user_state,
            journal,
            blocks,
            out,
        } => cmd_user_query(logn, srs, user_state, journal, blocks, out),

        Cmd::UserHistory {
            logn,
            srs,
            user_state,
            journal,
            start_block,
            blocks,
            out,
        } => cmd_user_history(logn, srs, user_state, journal, start_block, blocks, out),

        Cmd::DumpBlock {
            dataset_dir,
            dataset_segment_size,
            block,
        } => cmd_dump_block(dataset_dir, dataset_segment_size, block),
    }
}

fn cmd_dump_block(dataset_dir: PathBuf, dataset_segment_size: u32, block: u64) -> Result<()> {
    use buvc_rs::dataset::DatasetReader;

    let mut ds = DatasetReader::new(&dataset_dir, dataset_segment_size);
    let entries = ds.get_block(block as u32)?;

    println!("Block {}: {} modified accounts", block, entries.len());
    for (i, e) in entries.iter().enumerate() {
        println!(
            "#{:4} addr={:#x}  balance_wei_dec={}  balance_wei_hex=0x{:x}",
            i,
            e.address,
            e.balance, // decimal U256
            e.balance, // hex U256
        );
    }

    Ok(())
}

fn cmd_build_universe(
    dataset_dir: PathBuf,
    segment_size: u32,
    start_block: u64,
    end_block: u64,
    out: PathBuf,
) -> Result<()> {
    let mut dataset = buvc_rs::dataset::DatasetReader::new(&dataset_dir, segment_size);
    let mut addresses: HashSet<Address> = HashSet::new();

    eprintln!(
        "Building universe from blocks {} to {}...",
        start_block, end_block
    );
    dataset.iterate_range(start_block as u32, end_block as u32, |block, entries| {
        for entry in entries {
            addresses.insert(entry.address);
        }
        if block % 10_000 == 0 {
            eprintln!("  block {}, unique addrs {}", block, addresses.len());
        }
        Ok(())
    })?;

    let mut sorted_addrs: Vec<Address> = addresses.into_iter().collect();
    sorted_addrs.sort();

    let content = sorted_addrs
        .iter()
        .map(|a| format!("{:#x}\n", a))
        .collect::<String>();
    fs::write(&out, content)?;
    eprintln!(
        "Universe: {} addrs -> {}",
        sorted_addrs.len(),
        out.display()
    );
    Ok(())
}

/// Publisher snapshot: build vector v at block B0, compute commitment C_{B0}, write snapshot.json.
/// Paper-faithful: snapshot contains ONLY commitment (no witnesses).
fn cmd_publisher_snapshot(
    logn: usize,
    srs: PathBuf,
    block: u64,
    universe_file: PathBuf,
    snapshot_state: PathBuf,
    out: PathBuf,
) -> Result<()> {
    let n = 1usize << logn;

    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    let ctx = make_ctx(&vp, logn);

    let universe_addrs = read_addresses_file(&universe_file)?;
    eprintln!("[PublisherSnapshot] universe_addrs = {}, N = {}", universe_addrs.len(), n);
    let indexer = Indexer::from_universe_sequential(&universe_addrs, n)?;
    eprintln!("[PublisherSnapshot] Indexer ready (sequential mapping).");

    let bytes = fs::read(&snapshot_state)?;
    let st: SnapshotState = serde_json::from_slice(&bytes)?;
    eprintln!("[PublisherSnapshot] snapshot_state balances = {}", st.balances.len());
    if st.block_number != block {
        bail!(
            "snapshot_state block {} != requested snapshot block {}",
            st.block_number,
            block
        );
    }
    if st.balance_encoding != "u256" {
        bail!("snapshot_state.balance_encoding must be u256");
    }

    eprintln!("[PublisherSnapshot] Building state vector v for {} addresses at block {}", universe_addrs.len(), block);
    let mut v = vec![Fr::zero(); n];
    for &a in &universe_addrs {
        let key = format!("{:#x}", a);
        let bal_dec = st
            .balances
            .get(&key)
            .ok_or_else(|| eyre::eyre!("snapshot_state missing universe address {}", key))?;
        let bal_u = U256::from_dec_str(bal_dec)?;
        let bal_f = fr_from_u256_exact(bal_u)?;
        let i = indexer.index_of(a)?;
        v[i] = bal_f;
    }
    eprintln!("[PublisherSnapshot] State vector built, starting initial commitment generation...");

    let t0 = t_start();
    let (gc, _gq_all) = ctx.build_commitment(&v);
    emit("Commit", block, n, 0, 0, t_us(t0));
    eprintln!("[PublisherSnapshot] Initial commitment generation completed at block {}", block);

    let snap = SnapshotOut {
        block_number: block,
        n,
        logn,
        srs_id,
        gc_hex: g1_to_hex(&gc),
        universe_mode: "hashed_keccak_mod_n_collision_checked".into(),
        balance_encoding: "fr_exact_from_u256".into(),
    };

    fs::write(&out, serde_json::to_vec_pretty(&snap)?)?;
    eprintln!("publisher snapshot @{} written {}", block, out.display());
    Ok(())
}

/// Publisher advance: publishes per-block (β,Δ) and pinned commitments C_t in journal.
fn cmd_publisher_advance(
    logn: usize,
    srs: PathBuf,
    dataset_dir: PathBuf,
    dataset_segment_size: u32,
    universe_file: PathBuf,
    snapshot: PathBuf,
    snapshot_state: PathBuf,
    end_block: u64,
    journal_path: PathBuf,
) -> Result<()> {
    let n = 1usize << logn;

    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    let ctx = make_ctx(&vp, logn);

    let snap: SnapshotOut = serde_json::from_slice(&fs::read(&snapshot)?)?;
    if snap.logn != logn || snap.n != n || snap.srs_id != srs_id {
        bail!("snapshot invariants mismatch");
    }

    let universe_addrs = read_addresses_file(&universe_file)?;
    let indexer = Indexer::from_universe_sequential(&universe_addrs, n)?;
    let universe_set: HashSet<Address> = universe_addrs.iter().copied().collect();

    eprintln!("[PublisherAdvance] Starting: snapshot block {}, target end_block {}, journal: {}", snap.block_number, end_block, journal_path.display());
    let mut gc_cur = g1_from_hex(&snap.gc_hex)?;
    let mut cur_block = snap.block_number;
    if end_block < cur_block {
        bail!("end_block < snapshot block");
    }

    // Publisher needs exact values to compute deltas from dataset (application layer)
    eprintln!("[PublisherAdvance] Initializing state tracker from snapshot...");
    let mut tracker = StateTracker::from_snapshot_file(
        dataset_dir.as_path(),
        dataset_segment_size,
        snapshot_state.as_path(),
        universe_set,
    )?;
    if tracker.cur_block != cur_block {
        bail!(
            "snapshot_state block {} != snapshot block {}",
            tracker.cur_block,
            cur_block
        );
    }
    eprintln!("[PublisherAdvance] State tracker initialized, processing blocks from {} to {}...", cur_block, end_block);

    while cur_block < end_block {
        let changes = tracker.apply_next_block()?;
        cur_block = tracker.cur_block;

        let mut beta = Vec::<usize>::new();
        let mut delta = Vec::<Fr>::new();

        for (addr, oldf, newf) in changes {
            let idx = indexer.index_of(addr)?;
            beta.push(idx);
            delta.push(delta_fr(newf, oldf));
        }

        let (beta, delta) = canonicalize_beta_delta(&beta, &delta);

        // publisher updates commitment (users will do witness updates)
        let (gc_new, _) = vupdate_step(&ctx, gc_cur, &[], Vec::new(), &beta, &delta, cur_block, n);
        gc_cur = gc_new;

        let j = JournalLine {
            block_number: cur_block,
            n,
            logn,
            srs_id: srs_id.clone(),
            changed_indices: beta,
            delta_hex: delta.iter().map(fr_to_hex).collect(),
            gc_hex: Some(g1_to_hex(&gc_cur)),
        };
        append_line(&journal_path, &j)?;

        if cur_block % 10_000 == 0 {
            eprintln!("PublisherAdvance at block {}", cur_block);
        }
    }

    eprintln!("PublisherAdvance done. journal={}", journal_path.display());
    Ok(())
}

/// Publisher issues initial user state for α at snapshot:
/// - computes v at snapshot
/// - computes witnesses gq for α (publisher can compute full and extract α; user stores only α)
fn cmd_issue_user_state(
    logn: usize,
    srs: PathBuf,
    snapshot: PathBuf,
    universe_file: PathBuf,
    snapshot_state: PathBuf,
    alpha_addresses: Option<String>,
    alpha_file: Option<PathBuf>,
    out: PathBuf,
) -> Result<()> {
    let n = 1usize << logn;

    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    let ctx = make_ctx(&vp, logn);

    let snap: SnapshotOut = serde_json::from_slice(&fs::read(&snapshot)?)?;
    if snap.logn != logn || snap.n != n || snap.srs_id != srs_id {
        bail!("snapshot invariants mismatch");
    }

    let universe_addrs = read_addresses_file(&universe_file)?;
    let indexer = Indexer::from_universe_sequential(&universe_addrs, n)?;

    // α chosen by user
    let alpha_addrs = if let Some(f) = alpha_file {
        read_addresses_file(&f)?
    } else if let Some(list) = alpha_addresses {
        parse_addr_list(&list)?
    } else {
        bail!("provide --alpha-addresses or --alpha-file");
    };
    let (alpha_indices, alpha_hex, alpha_addrs_sorted) =
        build_alpha_strict(&alpha_addrs, &indexer)?;

    // Build full v at snapshot (publisher has snapshot_state)
    let st: SnapshotState = serde_json::from_slice(&fs::read(&snapshot_state)?)?;
    if st.block_number != snap.block_number {
        bail!(
            "snapshot_state block {} != snapshot block {}",
            st.block_number,
            snap.block_number
        );
    }
    if st.balance_encoding != "u256" {
        bail!("snapshot_state.balance_encoding must be u256");
    }

    eprintln!("[IssueUserState] Building state vector v for {} addresses at block {}", universe_addrs.len(), snap.block_number);
    let mut v = vec![Fr::zero(); n];
    for &a in &universe_addrs {
        let key = format!("{:#x}", a);
        let bal_dec = st
            .balances
            .get(&key)
            .ok_or_else(|| eyre::eyre!("snapshot_state missing universe address {}", key))?;
        let bal_u = U256::from_dec_str(bal_dec)?;
        let bal_f = fr_from_u256_exact(bal_u)?;
        let i = indexer.index_of(a)?;
        v[i] = bal_f;
    }
    eprintln!("[IssueUserState] State vector built, starting initial commitment generation for {} alpha indices...", alpha_indices.len());

    // Commit + witnesses (publisher side). User will keep only α witnesses.
    let (gc, gq_all) = ctx.build_commitment(&v);
    eprintln!("[IssueUserState] Initial commitment generation completed, extracting witnesses for {} alpha indices", alpha_indices.len());

    // Ensure commitment equals snapshot commitment
    let gc_snap = g1_from_hex(&snap.gc_hex)?;
    if g1_to_hex(&gc) != g1_to_hex(&gc_snap) {
        bail!("publisher recomputed commitment != snapshot commitment; check inputs");
    }

    // Extract α values + α witnesses
    let mut alpha_vals = Vec::<Fr>::with_capacity(alpha_indices.len());
    let mut alpha_wit = Vec::<G1>::with_capacity(alpha_indices.len());
    for (&i, &a) in alpha_indices.iter().zip(alpha_addrs_sorted.iter()) {
        let key = format!("{:#x}", a);
        let bal_dec = st
            .balances
            .get(&key)
            .ok_or_else(|| eyre::eyre!("snapshot_state missing α address {}", key))?;
        let bal_u = U256::from_dec_str(bal_dec)?;
        alpha_vals.push(fr_from_u256_exact(bal_u)?);
        alpha_wit.push(gq_all[i]);
    }

    let user = UserState {
        n,
        logn,
        srs_id: srs_id.clone(),
        last_block: snap.block_number,
        gc_hex: snap.gc_hex.clone(),
        alpha_indices,
        alpha_addresses: alpha_hex,
        alpha_values_hex: alpha_vals.iter().map(fr_to_hex).collect(),
        alpha_witnesses_hex: alpha_wit.iter().map(g1_to_hex).collect(),
        universe_mode: snap.universe_mode.clone(),
        value_mode: "user_maintains_values_forward".into(),
        witness_mode: "user_maintains_witnesses_vupdate".into(),
    };

    fs::write(&out, serde_json::to_vec_pretty(&user)?)?;
    eprintln!(
        "Issued user_state @{} -> {}",
        snap.block_number,
        out.display()
    );
    Ok(())
}

/// User query: start from user_state.json (maintained α values+witnesses),
/// apply VUpdate forward for each requested block using journal,
/// then aggregate + verify.
fn cmd_user_query(
    logn: usize,
    srs: PathBuf,
    user_state_path: PathBuf,
    journal: PathBuf,
    mut blocks: Vec<u64>,
    out: PathBuf,
) -> Result<()> {
    eprintln!("[UserQuery] Starting query processing...");
    eprintln!("[UserQuery] Loading user state from {}", user_state_path.display());
    let user0: UserState = serde_json::from_slice(&fs::read(&user_state_path)?)?;

    if user0.logn != logn {
        bail!("user_state.logn {} != {}", user0.logn, logn);
    }
    let n = user0.n;

    eprintln!("[UserQuery] User state loaded: last_block={}, {} alpha indices, n={}", user0.last_block, user0.alpha_indices.len(), n);
    eprintln!("[UserQuery] Loading SRS...");
    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    if user0.srs_id != srs_id {
        bail!("user_state.srs_id mismatch");
    }
    let ctx = make_ctx(&vp, logn);
    eprintln!("[UserQuery] SRS loaded successfully");

    if blocks.is_empty() {
        bail!("--blocks required");
    }
    blocks.sort_unstable();
    blocks.dedup();
    eprintln!("[UserQuery] Query blocks: {:?} ({} unique blocks)", blocks, blocks.len());

    // paper-faithful: forward maintenance only
    if blocks[0] < user0.last_block {
        bail!(
            "forward-only maintenance: min query block {} < user_state.last_block {}",
            blocks[0],
            user0.last_block
        );
    }

    // Load journal lines
    eprintln!("[UserQuery] Loading journal from {}...", journal.display());
    let lines = iter_lines_filtered(&journal, &srs_id, n, logn)?;
    let by_block = map_by_block(lines);
    eprintln!("[UserQuery] Journal loaded: {} blocks available", by_block.len());

    // Parse maintained state
    eprintln!("[UserQuery] Initializing maintained state (commitment, {} alpha values, {} witnesses)...", user0.alpha_values_hex.len(), user0.alpha_witnesses_hex.len());
    let mut gc_cur = g1_from_hex(&user0.gc_hex)?;
    let alpha = user0.alpha_indices.clone();

    let mut alpha_pos: HashMap<usize, usize> = HashMap::new();
    for (p, &i) in alpha.iter().enumerate() {
        alpha_pos.insert(i, p);
    }

    let mut vals_cur = Vec::<Fr>::with_capacity(user0.alpha_values_hex.len());
    for hx in &user0.alpha_values_hex {
        vals_cur.push(fr_from_hex(hx)?);
    }

    let mut gq_alpha = Vec::<G1>::with_capacity(user0.alpha_witnesses_hex.len());
    for hx in &user0.alpha_witnesses_hex {
        gq_alpha.push(g1_from_hex(hx)?);
    }

    let mut cur_block = user0.last_block;

    let no_verify = std::env::var_os("CAUCHY_NO_VERIFY").is_some();
    let mut out_all: Vec<QueryAtOut> = Vec::with_capacity(blocks.len());

    eprintln!("[UserQuery] Starting processing of {} query targets, current block: {}", blocks.len(), cur_block);
    for &target in &blocks {
        // ⬇️ NEW: start end-to-end timer for this query target
        let t_total = t_start();
        let blocks_to_process = target.saturating_sub(cur_block);
        eprintln!("[UserQuery] === Processing query for block {} ({} blocks to advance from {}) ===", target, blocks_to_process, cur_block);
        
        for b in cur_block + 1..=target {
            if (b - cur_block) % 1000 == 0 || b == cur_block + 1 || b == target {
                eprintln!("[UserQuery] Processing block {}/{} (target: {})", b, target, target);
            }
            
            let j = by_block
                .get(&b)
                .ok_or_else(|| eyre::eyre!("missing journal block {}", b))?;

            let mut d = Vec::<Fr>::with_capacity(j.delta_hex.len());
            for hx in &j.delta_hex {
                d.push(fr_from_hex(hx)?);
            }
            let (beta, d) = canonicalize_beta_delta(&j.changed_indices, &d);

            // VUpdate step: update commitment and α-witnesses
            let (gc1, gq1) = vupdate_step(&ctx, gc_cur, &alpha, gq_alpha, &beta, &d, b, n);
            gc_cur = gc1;
            gq_alpha = gq1;

            // user maintains α-values forward
            apply_to_alpha_values(&alpha_pos, &mut vals_cur, &beta, &d);

            // check publisher pinned commitment
            if let Some(ref pinned) = j.gc_hex {
                if pinned != &g1_to_hex(&gc_cur) {
                    bail!("commitment mismatch at block {} (journal vs user)", b);
                }
            }
        }

        eprintln!("[UserQuery] Block {}: All VUpdate steps completed, starting proof aggregation...", target);
        let t_agg = t_start();
        let gq_agg = ctx.aggregate_proof(&alpha, &gq_alpha);
        let agg_time = t_us(t_agg);
        emit("Aggregate", target, n, alpha.len(), 0, agg_time);
        eprintln!("[UserQuery] Block {}: Proof aggregation completed ({} us)", target, agg_time);

        eprintln!("[UserQuery] Block {}: Starting verification...", target);
        let t_ver = t_start();
        let ok = if no_verify {
            eprintln!("[UserQuery] Block {}: Verification skipped (CAUCHY_NO_VERIFY set)", target);
            true
        } else {
            ctx.verify_multi(&vp, gc_cur, &alpha, &vals_cur, gq_agg)
        };
        let ver_time = t_us(t_ver);
        emit("VerifyMulti", target, n, alpha.len(), 0, ver_time);
        eprintln!("[UserQuery] Block {}: Verification completed ({} us, result: {})", target, ver_time, if ok { "OK" } else { "FAILED" });

        let mut single_proofs = Vec::<SingleProof>::with_capacity(alpha.len());
        for k in 0..alpha.len() {
            single_proofs.push(SingleProof {
                index: alpha[k],
                address: user0.alpha_addresses[k].clone(),
                value_hex: fr_to_hex(&vals_cur[k]),
                gq_hex: g1_to_hex(&gq_alpha[k]),
            });
        }

        out_all.push(QueryAtOut {
            query_block: target,
            n,
            logn,
            srs_id: srs_id.clone(),
            gc_hex: g1_to_hex(&gc_cur),
            indices: alpha.clone(),
            addresses: user0.alpha_addresses.clone(),
            values_hex: vals_cur.iter().map(fr_to_hex).collect(),
            single_proofs,
            aggregated_proof_hex: g1_to_hex(&gq_agg),
            verify_multi_ok: ok,
            engine: "paper_faithful_user_maintained_alpha_values_and_witnesses_vupdate".into(),
        });
        let total_time = t_us(t_total);
        emit(
            "UserQueryTotal",
            target,
            n,
            alpha.len(),
            0, // you can use blocks_in_range if you prefer
            total_time,
        );
        eprintln!("[UserQuery] Block {}: Query processing completed (total: {} us)", target, total_time);

        cur_block = target;
    }

    // Persist updated user state (so user truly "maintains over time")
    eprintln!("[UserQuery] Persisting updated user state (last_block: {})...", cur_block);
    let updated_user = UserState {
        last_block: cur_block,
        gc_hex: g1_to_hex(&gc_cur),
        alpha_values_hex: vals_cur.iter().map(fr_to_hex).collect(),
        alpha_witnesses_hex: gq_alpha.iter().map(g1_to_hex).collect(),
        ..user0
    };
    fs::write(&user_state_path, serde_json::to_vec_pretty(&updated_user)?)?;
    eprintln!("[UserQuery] User state persisted to {}", user_state_path.display());

    eprintln!("[UserQuery] Writing query results to {}...", out.display());
    fs::write(&out, serde_json::to_vec_pretty(&out_all)?)?;
    eprintln!(
        "[UserQuery] Query processing complete: answered {} blocks, updated user_state",
        out_all.len()
    );
    Ok(())
}

/// UserHistory: use stream-based VUpdate(S, α) to compute witnesses for α
/// at many historical blocks, starting from final proofs at user_state.last_block.
///
/// This returns *witnesses only*; values at those blocks are meant to be
/// fetched from the underlying blockchain / RPC as in the paper.
fn cmd_user_history(
    logn: usize,
    srs: PathBuf,
    user_state_path: PathBuf,
    journal: PathBuf,
    start_block: u64,
    mut blocks: Vec<u64>,
    out: PathBuf,
) -> Result<()> {
    let user: UserState = serde_json::from_slice(&fs::read(&user_state_path)?)?;

    if user.logn != logn {
        bail!("user_state.logn {} != {}", user.logn, logn);
    }
    let n = user.n;

    let (vp, srs_id) = load_or_create_srs(&srs, logn)?;
    if user.srs_id != srs_id {
        bail!("user_state.srs_id mismatch");
    }
    let ctx = make_ctx(&vp, logn);

    if blocks.is_empty() {
        bail!("--blocks required");
    }
    blocks.sort_unstable();
    blocks.dedup();

    let last_block = user.last_block;
    if start_block >= last_block {
        bail!(
            "start_block {} must be < user_state.last_block {}",
            start_block,
            last_block
        );
    }
    if blocks[0] <= start_block {
        bail!(
            "all history blocks must satisfy start_block < block; got min block {} <= start_block {}",
            blocks[0],
            start_block
        );
    }
    if blocks.last().copied().unwrap() > last_block {
        bail!(
            "history blocks must be ≤ user_state.last_block {}; got max {}",
            last_block,
            blocks.last().copied().unwrap()
        );
    }

    // Load all journal lines for this srs/n/logn once.
    let all_lines = iter_lines_filtered(&journal, &srs_id, n, logn)?;

    // Restrict to the [start_block+1 ..= last_block] window and map by block.
    let mut by_block: BTreeMap<u64, JournalLine> = BTreeMap::new();
    for j in &all_lines {
        if j.block_number > last_block || j.block_number <= start_block {
            continue;
        }
        by_block.insert(j.block_number, j.clone());
    }

    // α (universe subset) the user cares about, from user_state.
    let alpha_all = user.alpha_indices.clone();

    // Forward stream S: sequence of UPDATE and QUERY ops between start_block and last_block.
    // Each QUERY carries the α it is about (paper's VUpdate(S, α) semantics).
    let mut ops_fwd: Vec<HistoryOp> = Vec::new();
    let mut query_id_to_block: Vec<(usize, u64)> = Vec::new();
    let block_set: HashSet<u64> = blocks.iter().copied().collect();

    for b in start_block + 1..=last_block {
        let j = by_block
            .get(&b)
            .ok_or_else(|| eyre::eyre!("missing journal block {} in history window", b))?;

        if !j.changed_indices.is_empty() {
            let mut d = Vec::<Fr>::with_capacity(j.delta_hex.len());
            for hx in &j.delta_hex {
                d.push(fr_from_hex(hx)?);
            }
            let (beta, delta) = canonicalize_beta_delta(&j.changed_indices, &d);
            if !beta.is_empty() {
                ops_fwd.push(HistoryOp::Update { beta, delta });
            }
        }

        if block_set.contains(&b) {
            let qid = query_id_to_block.len();
            ops_fwd.push(HistoryOp::Query {
                query_id: qid,
                alpha: alpha_all.clone(),
            });
            query_id_to_block.push((qid, b));
        }
    }

    if query_id_to_block.is_empty() {
        bail!("no history queries fell inside (start_block, last_block]");
    }

    // Reverse and negate updates to form the stream S' expected by VUpdate(S, α).
    let mut ops_rev = ops_fwd;
    ops_rev.reverse();
    for op in ops_rev.iter_mut() {
        if let HistoryOp::Update { delta, .. } = op {
            for d in delta.iter_mut() {
                *d = -*d;
            }
        }
    }

    // Final witnesses at user.last_block for α.
    let mut gq_final_all = Vec::<G1>::with_capacity(user.alpha_witnesses_hex.len());
    for hx in &user.alpha_witnesses_hex {
        gq_final_all.push(g1_from_hex(hx)?);
    }

    // Run the history algorithm: for each QUERY in S' we get a set of witnesses
    // corresponding to the α attached to that query (here always alpha_all).
    let t_hist = t_start();
    let results = vupdate_history(&ctx, &alpha_all, &gq_final_all, &ops_rev);
    let hist_micros = t_us(t_hist);

    emit(
        "UserHistoryTotal",
        last_block, // we’re rewinding from here
        n,
        alpha_all.len(), // |α|
        blocks.len(),    // number of history queries answered
        hist_micros,
    );
    // Map query_id -> block number.
    let mut q_map: HashMap<usize, u64> = HashMap::new();
    for (qid, b) in query_id_to_block.iter().copied() {
        q_map.insert(qid, b);
    }

    #[derive(serde::Serialize)]
    struct HistoryProofOut {
        query_block: u64,
        n: usize,
        logn: usize,
        srs_id: String,
        gc_hex: String,
        indices: Vec<usize>,
        addresses: Vec<String>,
        witnesses_hex: Vec<String>,
        engine: String,
    }

    let mut out_vec = Vec::<HistoryProofOut>::new();
    for r in results {
        let block = *q_map
            .get(&r.query_id)
            .ok_or_else(|| eyre::eyre!("unknown query id {}", r.query_id))?;

        let gc_hex = pinned_gc_at(&all_lines, block)
            .ok_or_else(|| eyre::eyre!("missing pinned gc_hex at block {}", block))?;

        let witnesses_hex = r.proofs.iter().map(g1_to_hex).collect();

        out_vec.push(HistoryProofOut {
            query_block: block,
            n,
            logn,
            srs_id: srs_id.clone(),
            gc_hex,
            indices: r.indices.clone(), // α from the query itself
            addresses: user.alpha_addresses.clone(),
            witnesses_hex,
            engine: "paper_faithful_history_vupdate_witnesses_only".into(),
        });
    }

    // Sort by block for nicer output.
    out_vec.sort_by_key(|h| h.query_block);

    fs::write(&out, serde_json::to_vec_pretty(&out_vec)?)?;
    eprintln!(
        "UserHistory produced {} historical proof sets between {} and {}, starting from state at {}",
        out_vec.len(),
        start_block,
        last_block,
        last_block
    );
    Ok(())
}
