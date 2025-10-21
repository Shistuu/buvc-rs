// src/bin/cauchy_erigon_runner.rs
use std::{collections::BTreeSet, fs, path::PathBuf, str::FromStr};

use clap::Parser;
use ethers::providers::{Middleware, Provider};
use ethers::types::{Address, BlockId, BlockNumber, U256, U64, H256};

use ark_bls12_381::{Fr, G1Projective as G1};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use buvc_rs::vc_context::VcContext;
use buvc_rs::vc_parameter::VcParameter;
use rand::Rng;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value="/mydata/erigon/mainnet/erigon.ipc")]
    ipc: String,

    /// If set, use this block hash (0x…); otherwise pick a random recent block from the node.
    #[arg(long)]
    block_hash: Option<String>,

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

    /// log2(n) for Cauchy vector size (n must be >= multi_k)
    #[arg(long, default_value_t = 16usize)]
    logn: usize,

    /// SRS file to reuse/create (DEV ONLY). In production, pin a trusted SRS!
    #[arg(long, default_value="srs.bin")]
    srs: PathBuf,

    /// Output JSON file
    #[arg(long, default_value="proof_out.json")]
    out: PathBuf,
}

#[derive(serde::Serialize)]
struct SingleProof {
    index: usize,
    address: String,
    balance_wei: String,
    gq_hex: String,
    value_hex: String,
}

#[derive(serde::Serialize)]
struct UpdateResult {
    update_block: u64,
    commitment_updated_hex: String,
    updated_proofs: Vec<SingleProof>,
    verify_multi_updated_ok: bool,
}

#[derive(serde::Serialize)]
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

#[derive(serde::Serialize)]
struct ProofBundle {
    // base commitment
    block_number: u64,
    block_hash: String,
    n: usize,
    srs_id: String,
    gc_b: String,

    // what we proved
    indices: Vec<usize>,
    addresses: Vec<String>,
    balances_wei: Vec<String>,
    single_proofs: Vec<SingleProof>,
    aggregated_proof_hex: String,
    verify_multi_ok: bool,

    // updates
    update_same: Option<UpdateResult>,
    update_different: Option<UpdateDifferentResult>,
}

//Small helpers (encoding/decoding)
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

// DEV SRS: reuse or create local file. PRODUCTION: replace with pinned SRS.
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
    (format!("blake3:{}", hex::encode(blake3::hash(&buf).as_bytes())), vp)
}

// ----------------- main -----------------

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    // Step 0 (I/O): connect to Erigon/Geth over IPC
    let provider = Provider::connect_ipc(args.ipc.clone()).await?;

    // Step 1 (data): pick a block (given hash or random recent) — return hash, number, and the full block
    let (block_hash, block_number, block) = if let Some(hs) = args.block_hash.as_deref() {
        let bh: H256 = hs.parse()?;
        let blk = provider
            .get_block_with_txs(bh)
            .await?
            .ok_or_else(|| color_eyre::eyre::eyre!("block {} not found", hs))?;
        let bn = blk.number.unwrap_or(U64::zero()).as_u64();
        (bh, bn, blk)
    } else {
        let latest = provider.get_block_number().await?.as_u64();
        if latest <= 1 { color_eyre::eyre::bail!("node latest <= 1; cannot pick random block"); }
        let start = latest.saturating_sub(5000);
        let mut rng = rand::thread_rng();
        let rand_bn = rng.gen_range(start.max(1)..=latest);
        let bid: BlockId = BlockNumber::Number(rand_bn.into()).into();
        let blk = provider
            .get_block_with_txs(bid)
            .await?
            .ok_or_else(|| color_eyre::eyre::eyre!("block {} not found", rand_bn))?;
        let bh = blk.hash.ok_or_else(|| color_eyre::eyre::eyre!("block has no hash"))?;
        (bh, rand_bn, blk)
    };

    // We already have `block` with txs; no need to refetch.
    if block.transactions.is_empty() {
        color_eyre::eyre::bail!("picked block {} has no transactions", block_number);
    }

    // Step 2 (address selection): choose m addresses (alpha indices)
    let chosen_addrs: Vec<Address> = if let Some(list) = args.addresses.as_deref() {
        list.split(',').map(|s| Address::from_str(s.trim())).collect::<Result<_, _>>()?
    } else {
        let mut set = BTreeSet::<Address>::new();
        for tx in &block.transactions {
            if let Some(to) = tx.to { set.insert(to); } else { set.insert(tx.from); }
            if set.len() >= args.multi_k { break; }
        }
        if set.is_empty() { color_eyre::eyre::bail!("no addresses could be selected from block"); }
        set.into_iter().collect()
    };

    // Step 3 (vector construction): v[i] = balance(addr_i) @ block_number, others 0
    let n = 1usize << args.logn;
    if chosen_addrs.len() > n {
        color_eyre::eyre::bail!("multi_k/addresses ({}) exceed n={}", chosen_addrs.len(), n);
    }
    let bid: BlockId = BlockNumber::Number(block_number.into()).into();
    let mut balances = Vec::<U256>::with_capacity(chosen_addrs.len());
    for a in &chosen_addrs {
        balances.push(provider.get_balance(*a, Some(bid)).await?);
    }
    let mut v = vec![Fr::from(0u64); n];
    for (i, bal) in balances.iter().enumerate() {
        v[i] = fr_from_u256(*bal);
    }

    // Step 4 (SRS & Context):
    // - SRS gives [s^i]_1, [s^i]_2 (paper’s CRS)
    // - VcContext precomputes the Cauchy structure (FFT roots, L_i, L'_i)
    let (srs_id, vp) = load_or_create_srs(&args.srs, args.logn);
    let vc_c = VcContext::new(&vp, args.logn);

    // Step 5 (commitment & witnesses): 
    // - build_commitment does the Cauchy evaluation/interpolation via FFTs (poly.rs)
    // - returns gc (commitment) and gq[i] (single-point witness per index)
    let (gc, gq) = vc_c.build_commitment(&v);

    // Step 6 (proofs):
    // - Single-point proofs (verify) for indices 0..m-1
    // - Aggregated multi-point proof for the same indices (aggregate_proof + verify_multi)
    let m = chosen_addrs.len();
    let indices: Vec<usize> = (0..m).collect();
    let values: Vec<Fr> = v[..m].to_vec();

    for i in 0..m {
        assert!(vc_c.verify(&vp, gc, i, values[i], gq[i]), "single-point verify failed at {}", i);
    }
    let gq_agg = vc_c.aggregate_proof(&indices, &gq[..m]);
    let verify_multi_ok = vc_c.verify_multi(&vp, gc, &indices, &values, gq_agg);

    let single_proofs = (0..m).map(|i| SingleProof {
        index: i,
        address: format!("{:#x}", chosen_addrs[i]),
        balance_wei: balances[i].to_string(),
        gq_hex: g1_to_hex(&gq[i]),
        value_hex: fr_to_hex(&values[i]),
    }).collect::<Vec<_>>();

    // Step 7 (updates — SAME indices): 
    //   Using update_commitment and update_witnesses_batch_same (alpha == beta)
    let update_same = if let Some(up_blk) = args.update_block {
        let up_bid: BlockId = BlockNumber::Number(up_blk.into()).into();
        // new balances for SAME addresses
        let mut new_balances = Vec::<U256>::with_capacity(m);
        for a in &chosen_addrs { new_balances.push(provider.get_balance(*a, Some(up_bid)).await?); }
        let delta_values: Vec<Fr> = (0..m)
            .map(|i| fr_from_u256(new_balances[i]) - fr_from_u256(balances[i])).collect();

        // update commitment
        let mut gc_updated = gc;
        for i in 0..m {
            gc_updated = vc_c.update_commitment(gc_updated, i, delta_values[i]);
        }

        // update witnesses (SAME alpha = beta) – pass slice directly
        let gq_updated = vc_c.update_witnesses_batch_same(&indices, &gq[..m], &delta_values);

        // verify updated multi-proof
        let values_updated: Vec<Fr> = (0..m).map(|i| values[i] + delta_values[i]).collect();
        let gq_agg_updated = vc_c.aggregate_proof(&indices, &gq_updated);
        let verify_multi_updated_ok =
            vc_c.verify_multi(&vp, gc_updated, &indices, &values_updated, gq_agg_updated);

        Some(UpdateResult {
            update_block: up_blk,
            commitment_updated_hex: g1_to_hex(&gc_updated),
            updated_proofs: (0..m).map(|i| SingleProof {
                index: i,
                address: format!("{:#x}", chosen_addrs[i]),
                balance_wei: new_balances[i].to_string(),
                gq_hex: g1_to_hex(&gq_updated[i]),
                value_hex: fr_to_hex(&values_updated[i]),
            }).collect(),
            verify_multi_updated_ok,
        })
    } else { None };

    // Step 8 (updates — DIFFERENT indices): paper §5 (alpha != beta)
    //   Show updating the same proved witnesses when *some other* indices in the vector changed.
    let update_different = if args.update_different {
        let up_blk = args.update_block.ok_or_else(|| {
            color_eyre::eyre::eyre!("--update-different requires --update-block")
        })?;
        // collect a second disjoint set of up to m addresses from the block (beta set)
        let mut betas = Vec::<Address>::new();
        'outer: for tx in &block.transactions {
            let cand = tx.to.unwrap_or(tx.from);
            // skip addresses already in alpha
            if chosen_addrs.contains(&cand) { continue; }
            if betas.iter().any(|x| x == &cand) { continue; }
            betas.push(cand);
            if betas.len() >= m { break 'outer; }
        }
        if betas.is_empty() {
            None
        } else {
            let beta_len = betas.len();
            let up_bid: BlockId = BlockNumber::Number(up_blk.into()).into();

            // balances at base block for betas (old) and at update_block (new)
            let mut beta_old = Vec::<U256>::with_capacity(beta_len);
            let mut beta_new = Vec::<U256>::with_capacity(beta_len);
            for a in &betas {
                beta_old.push(provider.get_balance(*a, Some(bid)).await?);
                beta_new.push(provider.get_balance(*a, Some(up_bid)).await?);
            }
            // delta values to apply at beta indices (we map beta -> indices m..m+beta_len-1)
            let beta_indices: Vec<usize> = (0..beta_len).map(|j| m + j).collect();
            if m + beta_len > n {
                color_eyre::eyre::bail!("not enough n to place beta indices (need {}, have {})", m+beta_len, n);
            }
            let beta_deltas: Vec<Fr> = (0..beta_len)
                .map(|j| fr_from_u256(beta_new[j]) - fr_from_u256(beta_old[j]))
                .collect();

            // update commitment by applying deltas at *beta* positions
            let mut gc_updated = gc;
            for (k, &j) in beta_indices.iter().enumerate() {
                gc_updated = vc_c.update_commitment(gc_updated, j, beta_deltas[k]);
            }

            // update *alpha* witnesses given updates on *beta* (alpha != beta):
            let gq_alpha_after =
                vc_c.update_witnesses_batch_different(&indices, &gq[..m], &beta_indices, &beta_deltas);

            // sanity: recompute alpha values (unchanged), only witnesses/commitment changed
            let gq_agg_updated = vc_c.aggregate_proof(&indices, &gq_alpha_after);
            let verify_multi_updated_ok =
                vc_c.verify_multi(&vp, gc_updated, &indices, &values, gq_agg_updated);

            Some(UpdateDifferentResult {
                update_block: up_blk,
                beta_indices,
                beta_addresses: betas.iter().map(|a| format!("{:#x}", a)).collect(),
                beta_deltas_hex: beta_deltas.iter().map(fr_to_hex).collect(),
                updated_proofs_for_alpha: (0..m).map(|i| SingleProof {
                    index: i,
                    address: format!("{:#x}", chosen_addrs[i]),
                    balance_wei: balances[i].to_string(), // alpha values unchanged in this scenario
                    gq_hex: g1_to_hex(&gq_alpha_after[i]),
                    value_hex: fr_to_hex(&values[i]),
                }).collect(),
                commitment_updated_hex: g1_to_hex(&gc_updated),
                verify_multi_updated_ok,
            })
        }
    } else { None };

    // Step 9 (bundle & write)
    let bundle = ProofBundle {
        block_number,
        block_hash: format!("{:#x}", block_hash),
        n,
        srs_id,
        gc_b: g1_to_hex(&gc),
        indices: indices.clone(),
        addresses: chosen_addrs.iter().map(|a| format!("{:#x}", a)).collect(),
        balances_wei: balances.iter().map(|b| b.to_string()).collect(),
        single_proofs,
        aggregated_proof_hex: g1_to_hex(&gq_agg),
        verify_multi_ok,
        update_same,
        update_different,
    };

    fs::write(&args.out, serde_json::to_vec_pretty(&bundle)?)?;
    eprintln!(
        "Cauchy VC proof written to {}\n   m={} indices {:?}\n   verify_multi_ok={}",
        args.out.display(),
        m,
        &indices,
        verify_multi_ok
    );
    if let Some(up) = &bundle.update_same {
        eprintln!("   SAME-indices update @{}: verify_multi_updated_ok={}", up.update_block, up.verify_multi_updated_ok);
    }
    if let Some(up) = &bundle.update_different {
        eprintln!("   DIFFERENT-indices update @{}: verify_multi_updated_ok={}", up.update_block, up.verify_multi_updated_ok);
    }
    Ok(())
}
