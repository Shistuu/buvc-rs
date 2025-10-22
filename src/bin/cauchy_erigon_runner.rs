// src/bin/cauchy_erigon_runner.rs
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    str::FromStr,
};

use clap::Parser;
use color_eyre::eyre::{bail, eyre, Result};
use ethers::providers::{Middleware, Provider};
use ethers::types::{
    Address, Block as EthBlock, BlockId, BlockNumber, Transaction as EthTx, U256, U64, H256,
};

use ark_bls12_381::{Fr, G1Projective as G1};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use buvc_rs::vc_context::VcContext;
use buvc_rs::vc_parameter::VcParameter;
use rand::Rng;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "/mydata/erigon/mainnet/erigon.ipc")]
    ipc: String,

    /// If set, use this block hash (0x…); otherwise pick a random recent block from the node.
    #[arg(long)]
    block_hash: Option<String>,

    /// Prove every block number in [block-from .. block-to] (inclusive).
    /// Mutually exclusive with --block-hash. Enables per-block proof JSONs.
    #[arg(long)]
    block_from: Option<u64>,

    /// End of the inclusive range (requires --block-from).
    #[arg(long)]
    block_to: Option<u64>,

    /// When using range-per-block, write per-block JSONs here (default: current dir).
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Comma-separated addresses to prove (0x…). If omitted, we pick from the block’s txs (to→from).
    #[arg(long)]
    addresses: Option<String>,

    /// How many addresses to pick from the block when --addresses not given.
    #[arg(long, default_value_t = 4usize)]
    multi_k: usize,

    /// Also fetch balances at this later block and update witnesses (SAME indices case).
    #[arg(long)]
    update_block: Option<u64>,

    /// Additionally demo DIFFERENT-indices update (alpha != beta) from the same update block.
    /// We’ll pick a second, disjoint set of up to --multi-k addresses from the same block as beta.
    #[arg(long, default_value_t = false)]
    update_different: bool,

    /// Answer history query for α at a past block: rewind from current block_number to this.
    #[arg(long)]
    history_at: Option<u64>,

    /// log2(n) for Cauchy vector size (n must be >= multi_k + possible beta)
    #[arg(long, default_value_t = 16usize)]
    logn: usize,

    /// SRS file to reuse/create (DEV ONLY). In production, pin a trusted SRS!
    #[arg(long, default_value = "srs.bin")]
    srs: PathBuf,

    /// Output JSON file (single-block mode)
    #[arg(long, default_value = "proof_out.json")]
    out: PathBuf,

    /// If set together with --block-from/--block-to and --addresses,
    /// produce ONE extra JSON that contains time-series balances+proofs across the range.
    #[arg(long)]
    balances_out: Option<PathBuf>,
}

/* ---------- Output types ---------- */

#[derive(serde::Serialize, Clone)]
struct SingleProof {
    index: usize,
    address: String,
    balance_wei: String,
    gq_hex: String,
    value_hex: String,
}

#[derive(serde::Serialize, Clone)]
struct UpdateResult {
    update_block: u64,
    commitment_updated_hex: String,
    updated_proofs: Vec<SingleProof>,
    verify_multi_updated_ok: bool,
}

#[derive(serde::Serialize, Clone)]
struct UpdateDifferentResult {
    update_block: u64,
    beta_indices: Vec<usize>,
    beta_addresses: Vec<String>,
    /// These are the beta deltas (used to update alpha witnesses).
    beta_deltas_hex: Vec<String>,
    updated_proofs_for_alpha: Vec<SingleProof>,
    commitment_updated_hex: String,
    verify_multi_updated_ok: bool,
}

#[derive(serde::Serialize, Clone)]
struct HistoryResult {
    history_block: u64,
    commitment_hist_hex: String,
    single_proofs_hist: Vec<SingleProof>,
    aggregated_proof_hist_hex: String,
    verify_multi_hist_ok: bool,
}

#[derive(serde::Serialize)]
struct ProofBundle {
    // base commitment
    block_number: u64,
    block_hash: String,
    n: usize,
    srs_id: String,
    gc_b: String,

    // what we proved (α)
    indices: Vec<usize>,
    addresses: Vec<String>,
    balances_wei: Vec<String>,
    single_proofs: Vec<SingleProof>,
    aggregated_proof_hex: String,
    verify_multi_ok: bool,

    // query 2/3
    update_same: Option<UpdateResult>,
    update_different: Option<UpdateDifferentResult>,
    history_at: Option<HistoryResult>,
}

/* ---------- Helpers (encoding) ---------- */

fn fr_from_u256(x: U256) -> Fr {
    let mut be = [0u8; 32];
    x.to_big_endian(&mut be);
    be.reverse();
    Fr::from_le_bytes_mod_order(&be)
}
fn fr_to_hex(f: &Fr) -> String {
    let mut bytes = Vec::new();
    f.serialize_compressed(&mut bytes).unwrap();
    format!("0x{}", hex::encode(bytes))
}
fn g1_to_hex(g: &G1) -> String {
    let mut bytes = Vec::new();
    g.serialize_compressed(&mut bytes).unwrap();
    format!("0x{}", hex::encode(bytes))
}

/* ---------- SRS ---------- */

fn load_or_create_srs(path: &PathBuf, logn: usize) -> (String, VcParameter) {
    if path.exists() {
        let bytes = fs::read(path).expect("read srs");
        let vp = VcParameter::deserialize_compressed(&*bytes).expect("bad srs");
        let id = format!("blake3:{}", hex::encode(blake3::hash(&bytes).as_bytes()));
        return (id, vp);
    }
    let (_trap, vp) = VcParameter::new(logn, &mut ark_std::test_rng());
    let mut buf = Vec::new();
    vp.serialize_compressed(&mut buf).unwrap();
    fs::write(path, &buf).expect("write srs");
    (
        format!("blake3:{}", hex::encode(blake3::hash(&buf).as_bytes())),
        vp,
    )
}

/* ---------- RPC helpers ---------- */

async fn fetch_balances(
    provider: &Provider<ethers::providers::Ipc>,
    addrs: &[Address],
    block_num: u64,
) -> Result<Vec<U256>> {
    let bid: BlockId = BlockNumber::Number(block_num.into()).into();
    let mut v = Vec::with_capacity(addrs.len());
    for a in addrs {
        v.push(provider.get_balance(*a, Some(bid)).await?);
    }
    Ok(v)
}

/* ---------- Address selection from a block ---------- */

fn choose_addresses_from_block(block: &EthBlock<EthTx>, k: usize) -> Vec<Address> {
    let mut set = BTreeSet::<Address>::new();
    for tx in &block.transactions {
        if let Some(to) = tx.to {
            set.insert(to);
        } else {
            set.insert(tx.from);
        }
        if set.len() >= k {
            break;
        }
    }
    set.into_iter().collect()
}

/* ---------- Core Cauchy build/verify for α ---------- */

#[derive(Clone)]
struct BuiltAlpha {
    n: usize,
    indices: Vec<usize>,
    values: Vec<Fr>,
    balances_wei: Vec<U256>,
    single_proofs: Vec<SingleProof>,
    gq_vec: Vec<G1>,
    gq_agg: G1,
    gc: G1,
}

fn build_alpha_bundle(
    vc_c: &VcContext,
    vp: &VcParameter,
    n: usize,
    addrs: &[Address],
    balances: &[U256],
) -> BuiltAlpha {
    assert!(addrs.len() == balances.len());
    let m = addrs.len();
    let mut v = vec![Fr::from(0u64); n];
    for (i, bal) in balances.iter().enumerate() {
        v[i] = fr_from_u256(*bal);
    }
    let (gc, gq) = vc_c.build_commitment(&v);
    let indices: Vec<usize> = (0..m).collect();
    let values: Vec<Fr> = v[..m].to_vec();

    for i in 0..m {
        assert!(vc_c.verify(vp, gc, i, values[i], gq[i]), "single verify fail @{}", i);
    }
    let gq_agg = vc_c.aggregate_proof(&indices, &gq[..m]);
    let ok = vc_c.verify_multi(vp, gc, &indices, &values, gq_agg);
    assert!(ok, "multi verify failed for α");

    let single_proofs = (0..m)
        .map(|i| SingleProof {
            index: i,
            address: format!("{:#x}", addrs[i]),
            balance_wei: balances[i].to_string(),
            gq_hex: g1_to_hex(&gq[i]),
            value_hex: fr_to_hex(&values[i]),
        })
        .collect();

    BuiltAlpha {
        n,
        indices,
        values,
        balances_wei: balances.to_vec(),
        single_proofs,
        gq_vec: gq,
        gq_agg,
        gc,
    }
}

fn update_commitment_batch(vc_c: &VcContext, mut gc: G1, idx: &[usize], deltas: &[Fr]) -> G1 {
    assert_eq!(idx.len(), deltas.len());
    for (i, d) in idx.iter().zip(deltas.iter()) {
        gc = vc_c.update_commitment(gc, *i, *d);
    }
    gc
}

/* ---------- Query 2: Batch update SAME(α) ---------- */

fn do_update_same(
    vc_c: &VcContext,
    vp: &VcParameter,
    alpha: &BuiltAlpha,
    new_balances_alpha: &[U256], // balances of α at update_block
    update_block: u64,
) -> UpdateResult {
    let m = alpha.indices.len();
    let mut delta_values = Vec::with_capacity(m);
    let mut values_up = alpha.values.clone();

    for i in 0..m {
        let old = alpha.values[i];
        let new = fr_from_u256(new_balances_alpha[i]);
        let d = new - old;
        delta_values.push(d);
        values_up[i] = new;
    }

    let gq_updated =
        vc_c.update_witnesses_batch_same(&alpha.indices, &alpha.gq_vec[..m], &delta_values);
    let gc_updated = update_commitment_batch(vc_c, alpha.gc, &alpha.indices, &delta_values);

    let gq_agg_up = vc_c.aggregate_proof(&alpha.indices, &gq_updated);
    let verify_multi_updated_ok =
        vc_c.verify_multi(vp, gc_updated, &alpha.indices, &values_up, gq_agg_up);

    let updated_proofs = (0..m)
        .map(|i| SingleProof {
            index: i,
            address: alpha.single_proofs[i].address.clone(),
            balance_wei: new_balances_alpha[i].to_string(),
            gq_hex: g1_to_hex(&gq_updated[i]),
            value_hex: fr_to_hex(&values_up[i]),
        })
        .collect();

    UpdateResult {
        update_block,
        commitment_updated_hex: g1_to_hex(&gc_updated),
        updated_proofs,
        verify_multi_updated_ok,
    }
}

/* ---------- Query 2: Batch update DIFFERENT(β) ---------- */

fn do_update_different(
    vc_c: &VcContext,
    vp: &VcParameter,
    alpha: &BuiltAlpha,
    beta_indices: &[usize], // positions of β in the same size-n vector
    beta_deltas: &[Fr],     // Δ on β relative to current α-state
    beta_addresses: &[Address],
    update_block: u64,
) -> UpdateDifferentResult {
    let m = alpha.indices.len();
    let gq_alpha_updated = vc_c.update_witnesses_batch_different(
        &alpha.indices,
        &alpha.gq_vec[..m],
        beta_indices,
        beta_deltas,
    );
    let gc_diff_updated = update_commitment_batch(vc_c, alpha.gc, beta_indices, beta_deltas);

    // α values unchanged in DIFFERENT update
    let gq_agg_alpha_up = vc_c.aggregate_proof(&alpha.indices, &gq_alpha_updated);
    let verify_multi_updated_ok =
        vc_c.verify_multi(vp, gc_diff_updated, &alpha.indices, &alpha.values, gq_agg_alpha_up);

    UpdateDifferentResult {
        update_block,
        beta_indices: beta_indices.to_vec(),
        beta_addresses: beta_addresses
            .iter()
            .map(|a| format!("{:#x}", a))
            .collect(),
        beta_deltas_hex: beta_deltas.iter().map(|d| fr_to_hex(d)).collect(),
        updated_proofs_for_alpha: (0..m)
            .map(|i| SingleProof {
                index: i,
                address: alpha.single_proofs[i].address.clone(),
                balance_wei: alpha.balances_wei[i].to_string(),
                gq_hex: g1_to_hex(&gq_alpha_updated[i]),
                value_hex: fr_to_hex(&alpha.values[i]),
            })
            .collect(),
        commitment_updated_hex: g1_to_hex(&gc_diff_updated),
        verify_multi_updated_ok,
    }
}

/* ---------- Query 3: History-at (rewind via −Δ on β) ---------- */

async fn do_history_at(
    provider: &Provider<ethers::providers::Ipc>,
    vc_c: &VcContext,
    vp: &VcParameter,
    alpha: BuiltAlpha,
    block_now: u64,
    block_hist: u64,
    // mapping for touched addresses → vector indices; we’ll place any “new β” after α
    mut addr_to_index: BTreeMap<Address, usize>,
    n: usize,
) -> Result<HistoryResult> {
    if block_hist >= block_now {
        bail!("history_at must be < current block");
    }

    let mut next_free = alpha.indices.len(); // next free slot in [α..)
    let mut gc_hist = alpha.gc;
    let mut gq_hist = alpha.gq_vec[..alpha.indices.len()].to_vec();
    let mut values_hist = alpha.values.clone();

    let mut t = block_now;
    while t > block_hist {
        let bid_t: BlockId = BlockNumber::Number(t.into()).into();
        let blk_t = provider
            .get_block_with_txs(bid_t)
            .await?
            .ok_or_else(|| eyre!("block {} not found during history", t))?;

        // collect touched addresses at t
        let mut touched = BTreeSet::<Address>::new();
        for tx in &blk_t.transactions {
            if let Some(to) = tx.to {
                touched.insert(to);
            }
            touched.insert(tx.from);
        }
        if touched.is_empty() {
            t -= 1;
            continue;
        }

        // map touched → β indices; assign new slots as needed
        let mut beta_indices: Vec<usize> = Vec::with_capacity(touched.len());
        let mut beta_addrs: Vec<Address> = Vec::with_capacity(touched.len());
        for a in touched {
            let idx = *addr_to_index.entry(a).or_insert_with(|| {
                let i = next_free;
                next_free += 1;
                i
            });
            if idx >= n {
                bail!("n too small to place β during history rewind");
            }
            beta_indices.push(idx);
            beta_addrs.push(a);
        }

        // fetch balances at t and t-1 for β, compute forward Δₜ, then apply −Δₜ
        let balances_t = fetch_balances(provider, &beta_addrs, t).await?;
        let balances_t_1 = fetch_balances(provider, &beta_addrs, t - 1).await?;
        let mut deltas: Vec<Fr> = Vec::with_capacity(beta_indices.len());
        for i in 0..beta_indices.len() {
            let old = fr_from_u256(balances_t_1[i]);
            let new = fr_from_u256(balances_t[i]);
            deltas.push(new - old); // forward Δₜ
        }
        // apply −Δₜ to commitment and α-witnesses
        let neg: Vec<Fr> = deltas.iter().map(|d| -*d).collect();
        gq_hist = vc_c.update_witnesses_batch_different(
            &alpha.indices,
            &gq_hist,
            &beta_indices,
            &neg,
        );
        gc_hist = update_commitment_batch(vc_c, gc_hist, &beta_indices, &neg);

        // if α overlaps β, update values_hist[i] -= Δₜ
        for (k, bi) in beta_indices.iter().enumerate() {
            if *bi < alpha.indices.len() {
                values_hist[*bi] -= deltas[k];
            }
        }

        t -= 1;
    }

    let gq_agg_hist = vc_c.aggregate_proof(&alpha.indices, &gq_hist);
    let verify_multi_hist_ok =
        vc_c.verify_multi(vp, gc_hist, &alpha.indices, &values_hist, gq_agg_hist);

    let single_proofs_hist = (0..alpha.indices.len())
        .map(|i| SingleProof {
            index: i,
            address: alpha.single_proofs[i].address.clone(),
            balance_wei: "HIST".into(), // optional: fetch exact wei at block_hist for α
            gq_hex: g1_to_hex(&gq_hist[i]),
            value_hex: fr_to_hex(&values_hist[i]),
        })
        .collect();

    Ok(HistoryResult {
        history_block: block_hist,
        commitment_hist_hex: g1_to_hex(&gc_hist),
        single_proofs_hist,
        aggregated_proof_hist_hex: g1_to_hex(&gq_agg_hist),
        verify_multi_hist_ok,
    })
}

/* ---------- Build bundle for a given block (queries 1 & 4 always; 2/3 optional) ---------- */

async fn prove_one_block_bundle(
    provider: &Provider<ethers::providers::Ipc>,
    args: &Args,
    vp: &VcParameter,
    vc_c: &VcContext,
    block_number: u64,
    block_hash: H256,
    block: EthBlock<EthTx>,
) -> Result<ProofBundle> {
    // Address selection
    let chosen_addrs: Vec<Address> = if let Some(list) = args.addresses.as_deref() {
        list.split(',')
            .map(|s| Address::from_str(s.trim()))
            .collect::<Result<_, _>>()?
    } else {
        let picked = choose_addresses_from_block(&block, args.multi_k);
        if picked.is_empty() {
            bail!("no addresses could be selected from block");
        }
        picked
    };

    // Vector size / balances at block_number
    let n = 1usize << args.logn;
    if chosen_addrs.len() > n {
        bail!(
            "multi_k/addresses ({}) exceed n={}",
            chosen_addrs.len(),
            n
        );
    }

    let balances = fetch_balances(provider, &chosen_addrs, block_number).await?;
    let alpha = build_alpha_bundle(vc_c, vp, n, &chosen_addrs, &balances);

    // Base (1: value proofs, 4: aggregation)
    let mut bundle = ProofBundle {
        block_number,
        block_hash: format!("{:#x}", block_hash),
        n,
        srs_id: String::new(), // filled by caller
        gc_b: g1_to_hex(&alpha.gc),
        indices: alpha.indices.clone(),
        addresses: chosen_addrs.iter().map(|a| format!("{:#x}", a)).collect(),
        balances_wei: alpha.balances_wei.iter().map(|b| b.to_string()).collect(),
        single_proofs: alpha.single_proofs.clone(),
        aggregated_proof_hex: g1_to_hex(&alpha.gq_agg),
        verify_multi_ok: true,
        update_same: None,
        update_different: None,
        history_at: None,
    };

    // Query 2 (SAME α) if requested
    if let Some(up_bn) = args.update_block {
        let new_balances_alpha = fetch_balances(provider, &chosen_addrs, up_bn).await?;
        let upd_same = do_update_same(vc_c, vp, &alpha, &new_balances_alpha, up_bn);
        bundle.update_same = Some(upd_same);

        // Query 2 (DIFFERENT β) if requested
        if args.update_different {
            // pick disjoint β from same block’s txs (simple heuristic)
            let mut beta_addrs = Vec::<Address>::new();
            'txs: for tx in &block.transactions {
                if let Some(to) = tx.to {
                    if !chosen_addrs.contains(&to) && !beta_addrs.contains(&to) {
                        beta_addrs.push(to);
                        if beta_addrs.len() >= args.multi_k {
                            break 'txs;
                        }
                    }
                }
                let f = tx.from;
                if !chosen_addrs.contains(&f) && !beta_addrs.contains(&f) {
                    beta_addrs.push(f);
                    if beta_addrs.len() >= args.multi_k {
                        break 'txs;
                    }
                }
            }
            if !beta_addrs.is_empty() {
                // map β after α in the vector
                let beta_indices: Vec<usize> =
                    (alpha.indices.len()..alpha.indices.len() + beta_addrs.len()).collect();
                if *beta_indices.last().unwrap() >= n {
                    bail!("n too small to place β for DIFFERENT update");
                }
                // compute Δβ between current block_number and up_bn
                let beta_now = fetch_balances(provider, &beta_addrs, block_number).await?;
                let beta_up = fetch_balances(provider, &beta_addrs, up_bn).await?;
                let beta_deltas: Vec<Fr> = beta_now
                    .iter()
                    .zip(beta_up.iter())
                    .map(|(b_now, b_up)| fr_from_u256(*b_up) - fr_from_u256(*b_now))
                    .collect();

                let upd_diff = do_update_different(
                    vc_c,
                    vp,
                    &alpha,
                    &beta_indices,
                    &beta_deltas,
                    &beta_addrs,
                    up_bn,
                );
                bundle.update_different = Some(upd_diff);
            }
        }
    }

    // Query 3 (History-at t) if requested
    if let Some(hist_bn) = args.history_at {
        if hist_bn < block_number {
            // address→index map: α first, β grows after α as we encounter touched addrs
            let mut addr_to_index = BTreeMap::<Address, usize>::new();
            for (i, a) in chosen_addrs.iter().enumerate() {
                addr_to_index.insert(*a, i);
            }

            let hist = do_history_at(
                provider, vc_c, vp, alpha.clone(), block_number, hist_bn, addr_to_index, n,
            )
            .await?;
            bundle.history_at = Some(hist);
        }
    }

    Ok(bundle)
}

/* ----------------- main ----------------- */

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();
    let provider = Provider::connect_ipc(args.ipc.clone()).await?;

    /* ---- Range-per-block mode (existing) ---- */
    if let (Some(from), Some(to)) = (args.block_from, args.block_to) {
        if args.block_hash.is_some() {
            bail!("--block-hash cannot be used with --block-from/--block-to");
        }
        if to < from {
            bail!("--block-to must be >= --block-from");
        }

        let out_dir = args.out_dir.clone().unwrap_or_else(|| PathBuf::from("."));
        fs::create_dir_all(&out_dir)?;

        let (srs_id, vp) = load_or_create_srs(&args.srs, args.logn);
        let vc_c = VcContext::new(&vp, args.logn);

        // optional: for 5th query (balances across range)
        let mut range_series: Vec<(u64, ProofBundle)> = Vec::new();

        for bn in from..=to {
            let bid: BlockId = BlockNumber::Number(bn.into()).into();
            let Some(blk) = provider.get_block_with_txs(bid).await? else {
                eprintln!("[{}] not found; skipping", bn);
                continue;
            };
            if blk.transactions.is_empty() {
                eprintln!("[{}] has no transactions; skipping", bn);
                continue;
            }
            let Some(bh) = blk.hash else {
                eprintln!("[{}] has no hash; skipping", bn);
                continue;
            };

            let mut bundle =
                prove_one_block_bundle(&provider, &args, &vp, &vc_c, bn, bh, blk).await?;
            bundle.srs_id = srs_id.clone();

            let out_path = out_dir.join(format!("proof_{}.json", bn));
            fs::write(&out_path, serde_json::to_vec_pretty(&bundle)?)?;
            eprintln!(
                "Cauchy VC proof written to {}\n   m={} indices {:?}\n   verify_multi_ok={}",
                out_path.display(),
                bundle.indices.len(),
                &bundle.indices,
                bundle.verify_multi_ok
            );

            // collect for 5th query only if user provided explicit addresses (fixed α)
            if args.addresses.is_some() {
                range_series.push((bn, bundle));
            }
        }

        // ----- 5) Range balances query (single JSON) -----
        // If user supplied addresses and balances_out, emit time-series JSON
        if let (Some(_addr_str), Some(out_path)) = (&args.addresses, &args.balances_out) {
            // shape: for each address in α, a list of (block, balance_wei), plus per-block aggregated proof & commitment
            #[derive(serde::Serialize)]
            struct AddrSeries {
                address: String,
                points: Vec<(u64, String)>,
            }
            #[derive(serde::Serialize)]
            struct RangeBalances {
                from: u64,
                to: u64,
                n: usize,
                srs_id: String,
                alpha_indices: Vec<usize>,
                series: Vec<AddrSeries>,
                per_block: Vec<serde_json::Value>, // each bundle’s {block_number, gc_b, aggregated_proof_hex}
            }

            if let Some((_bn0, first_bundle)) = range_series.first() {
                let m = first_bundle.indices.len();
                // pivot balances
                let mut per_addr: Vec<AddrSeries> = (0..m)
                    .map(|i| AddrSeries {
                        address: first_bundle.addresses[i].clone(),
                        points: Vec::new(),
                    })
                    .collect();

                for (bn, b) in &range_series {
                    for i in 0..m {
                        per_addr[i].points.push((*bn, b.balances_wei[i].clone()));
                    }
                }

                let per_block_meta: Vec<serde_json::Value> = range_series
                    .iter()
                    .map(|(bn, b)| {
                        serde_json::json!({
                            "block_number": bn,
                            "gc_b": b.gc_b,
                            "aggregated_proof_hex": b.aggregated_proof_hex,
                            "verify_multi_ok": b.verify_multi_ok
                        })
                    })
                    .collect();

                let payload = RangeBalances {
                    from,
                    to,
                    n: first_bundle.n,
                    srs_id: first_bundle.srs_id.clone(),
                    alpha_indices: first_bundle.indices.clone(),
                    series: per_addr,
                    per_block: per_block_meta,
                };
                fs::write(out_path, serde_json::to_vec_pretty(&payload)?)?;
                eprintln!(
                    "Range balances (Cauchy-backed) written to {}",
                    out_path.display()
                );
            }
        }

        return Ok(());
    }

    /* ---- Single-block mode ---- */

    // pick block (hash or random recent)
    let (block_hash, block_number, block) = if let Some(hs) = args.block_hash.as_deref() {
        let bh: H256 = hs.parse()?;
        let blk = provider
            .get_block_with_txs(bh)
            .await?
            .ok_or_else(|| eyre!("block {} not found", hs))?;
        let bn = blk.number.unwrap_or(U64::zero()).as_u64();
        (bh, bn, blk)
    } else {
        let latest = provider.get_block_number().await?.as_u64();
        if latest <= 1 {
            bail!("node latest <= 1; cannot pick random block");
        }
        let start = latest.saturating_sub(5000);
        let mut rng = rand::thread_rng();
        let rand_bn = rng.gen_range(start.max(1)..=latest);
        let bid: BlockId = BlockNumber::Number(rand_bn.into()).into();
        let blk = provider
            .get_block_with_txs(bid)
            .await?
            .ok_or_else(|| eyre!("block {} not found", rand_bn))?;
        let bh = blk.hash.ok_or_else(|| eyre!("block has no hash"))?;
        (bh, rand_bn, blk)
    };

    if block.transactions.is_empty() {
        bail!("picked block {} has no transactions", block_number);
    }

    // SRS/context
    let (srs_id, vp) = load_or_create_srs(&args.srs, args.logn);
    let vc_c = VcContext::new(&vp, args.logn);

    // Build all requested queries
    let mut bundle =
        prove_one_block_bundle(&provider, &args, &vp, &vc_c, block_number, block_hash, block)
            .await?;
    bundle.srs_id = srs_id;

    // Write
    fs::write(&args.out, serde_json::to_vec_pretty(&bundle)?)?;
    eprintln!(
        "Cauchy VC proof written to {}\n   m={} indices {:?}\n   verify_multi_ok={}",
        args.out.display(),
        bundle.indices.len(),
        &bundle.indices,
        bundle.verify_multi_ok
    );
    if let Some(up) = &bundle.update_same {
        eprintln!(
            "   SAME-indices update @{}: verify_multi_updated_ok={}",
            up.update_block, up.verify_multi_updated_ok
        );
    }
    if let Some(up) = &bundle.update_different {
        eprintln!(
            "   DIFFERENT-indices update @{}: verify_multi_updated_ok={}",
            up.update_block, up.verify_multi_updated_ok
        );
    }
    if let Some(h) = &bundle.history_at {
        eprintln!(
            "   HISTORY-at {}: verify_multi_hist_ok={}",
            h.history_block, h.verify_multi_hist_ok
        );
    }
    Ok(())
}
