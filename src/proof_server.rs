// src/proof_server.rs

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use serde::{Serialize, Deserialize};
use eyre::{Result, bail};

use ark_bls12_381::{G1Projective as G1, fr::Fr};
use ark_ff::{Field, Zero};

use crate::codec::{g1_from_hex, g1_to_hex, fr_from_hex, fr_to_hex};
use crate::types::{UserState, JournalLine};
use crate::vc_context::VcContext;
use crate::history::{HistoryOp, vupdate_history_same_alpha};
use crate::journal::for_each_line_filtered;

/* ============================================================
 * Helpers
 * ============================================================ */

fn t_start() -> Instant {
    Instant::now()
}

fn t_us(t0: Instant) -> u128 {
    t0.elapsed().as_micros() as u128
}

pub fn canonicalize_beta_delta(
    beta: &[usize],
    delta: &[Fr],
) -> (Vec<usize>, Vec<Fr>) {
    use std::collections::BTreeMap;

    let mut acc: BTreeMap<usize, Fr> = BTreeMap::new();
    for (&i, &d) in beta.iter().zip(delta.iter()) {
        if !d.is_zero() {
            *acc.entry(i).or_insert(Fr::ZERO) += d;
        }
    }
    acc.into_iter().unzip()
}

/* ============================================================
 * Proof-server data structures
 * ============================================================ */

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofServerUser {
    pub user_id: String,
    pub alpha_indices: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofServerState {
    pub n: usize,
    pub logn: usize,
    pub srs_id: String,

    /// Latest block server is synced to
    pub last_block: u64,

    /// Commitment at last_block
    pub gc_hex: String,

    /// UNION α across all users
    pub alpha_indices: Vec<usize>,

    /// Witnesses for UNION α at last_block
    pub alpha_witnesses_hex: Vec<String>,

    /// Values for UNION α at last_block (hex-encoded Fr)
    pub alpha_values_hex: Vec<String>,

    pub users: Vec<ProofServerUser>,
}

/* ============================================================
 * Construction
 * ============================================================ */

impl ProofServerState {
    pub fn from_user_states(
        users: &[(String, UserState)],
    ) -> Result<Self> {
        if users.is_empty() {
            bail!("ProofServerInit: no users");
        }

        let base = &users[0].1;

        let mut union_alpha = Vec::<usize>::new();
        let mut wit_map: HashMap<usize, String> = HashMap::new();
        let mut val_map: HashMap<usize, String> = HashMap::new();

        for (_, u) in users {
            if u.n != base.n || u.logn != base.logn || u.srs_id != base.srs_id {
                bail!("user_state invariant mismatch");
            }
            if u.last_block != base.last_block || u.gc_hex != base.gc_hex {
                bail!("user_states not aligned at same block");
            }

            for (i, &idx) in u.alpha_indices.iter().enumerate() {
                union_alpha.push(idx);
                wit_map
                    .entry(idx)
                    .or_insert_with(|| u.alpha_witnesses_hex[i].clone());
                val_map
                    .entry(idx)
                    .or_insert_with(|| u.alpha_values_hex[i].clone());
            }
        }

        union_alpha.sort_unstable();
        union_alpha.dedup();

        let mut alpha_witnesses_hex = Vec::with_capacity(union_alpha.len());
        let mut alpha_values_hex = Vec::with_capacity(union_alpha.len());
        for &idx in &union_alpha {
            alpha_witnesses_hex.push(
                wit_map
                    .get(&idx)
                    .ok_or_else(|| eyre::eyre!("missing witness {}", idx))?
                    .clone(),
            );
            alpha_values_hex.push(
                val_map
                    .get(&idx)
                    .ok_or_else(|| eyre::eyre!("missing value {}", idx))?
                    .clone(),
            );
        }

        let users_out = users
            .iter()
            .map(|(id, u)| ProofServerUser {
                user_id: id.clone(),
                alpha_indices: u.alpha_indices.clone(),
            })
            .collect();

        Ok(Self {
            n: base.n,
            logn: base.logn,
            srs_id: base.srs_id.clone(),
            last_block: base.last_block,
            gc_hex: base.gc_hex.clone(),
            alpha_indices: union_alpha,
            alpha_witnesses_hex,
            alpha_values_hex,
            users: users_out,
        })
    }

    /* ============================================================
     * Online forward maintenance
     * ============================================================ */

    pub fn apply_update(
        &mut self,
        ctx: &VcContext,
        beta: &[usize],
        delta: &[Fr],
        block: u64,
        pinned_gc: Option<&str>,
    ) -> Result<()> {
        let t0 = t_start();
        let mut gc = g1_from_hex(&self.gc_hex)?;

        // Update global commitment
        for (&i, &d) in beta.iter().zip(delta.iter()) {
            if !d.is_zero() {
                gc = ctx.update_commitment(gc, i, d);
            }
        }

        if let Some(p) = pinned_gc {
            if g1_to_hex(&gc) != p {
                bail!("commitment mismatch at block {}", block);
            }
        }

        // Update union-α witnesses if necessary
        if !beta.is_empty() && !self.alpha_indices.is_empty() {
            let mut gq = Vec::<G1>::with_capacity(self.alpha_witnesses_hex.len());
            for hx in &self.alpha_witnesses_hex {
                gq.push(g1_from_hex(hx)?);
            }

            gq = ctx.update_witnesses_batch(
                &self.alpha_indices,
                &gq,
                beta,
                delta,
            );

            self.alpha_witnesses_hex = gq.iter().map(g1_to_hex).collect();
        }

        // Update union-α values (server-tracked balances)
        if !beta.is_empty() && !self.alpha_indices.is_empty() {
            let mut pos = HashMap::<usize, usize>::new();
            for (i, &idx) in self.alpha_indices.iter().enumerate() {
                pos.insert(idx, i);
            }

            for (&b, &d) in beta.iter().zip(delta.iter()) {
                if let Some(&p) = pos.get(&b) {
                    let mut v = fr_from_hex(&self.alpha_values_hex[p])?;
                    v += d;
                    self.alpha_values_hex[p] = fr_to_hex(&v);
                }
            }
        }

        self.gc_hex = g1_to_hex(&gc);
        self.last_block = block;
        let dt = t_us(t0);
        eprintln!(
            "[ProofServerState::apply_update] block={} beta={} alpha={} micros={}",
            block,
            beta.len(),
            self.alpha_indices.len(),
            dt
        );
        Ok(())
    }

    /* ============================================================
     * Serve current head proofs
     * ============================================================ */

    pub fn serve_user(
        &self,
        user_id: &str,
    ) -> Option<(Vec<usize>, Vec<String>, Vec<String>)> {
        let user = self.users.iter().find(|u| u.user_id == user_id)?;
    
        let mut pos = HashMap::new();
        for (i, &idx) in self.alpha_indices.iter().enumerate() {
            pos.insert(idx, i);
        }
    
        let mut idxs = Vec::new();
        let mut wits = Vec::new();
        let mut vals = Vec::new();
    
        for &i in &user.alpha_indices {
            let p = *pos.get(&i)?;
            idxs.push(i);
            wits.push(self.alpha_witnesses_hex[p].clone());
            vals.push(self.alpha_values_hex[p].clone());
        }
    
        Some((idxs, vals, wits))
    }
    
    /// Return (user_alpha_indices, positions in union-α vector)
    pub fn user_alpha(
        &self,
        user_id: &str,
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let user = self.users.iter().find(|u| u.user_id == user_id)?;

        let mut pos = HashMap::<usize, usize>::new();
        for (i, &idx) in self.alpha_indices.iter().enumerate() {
            pos.insert(idx, i);
        }

        let mut positions = Vec::new();
        for &idx in &user.alpha_indices {
            let p = *pos.get(&idx)?;
            positions.push(p);
        }

        Some((user.alpha_indices.clone(), positions))
    }

    /* ============================================================
     * Historical range query (paper-faithful)
     * ============================================================ */

    pub fn history_query_same_alpha(
        &self,
        ctx: &VcContext,
        journal: &Path,
        start_block: u64,
        mut blocks: Vec<u64>,
    ) -> Result<Vec<(u64, Vec<G1>)>> {
        let t_total = t_start();
        blocks.sort_unstable();
        blocks.dedup();

        if blocks.is_empty() {
            bail!("no query blocks");
        }
        if start_block >= self.last_block {
            bail!("start_block >= last_block");
        }

        let mut gq_final = Vec::<G1>::new();
        for hx in &self.alpha_witnesses_hex {
            gq_final.push(g1_from_hex(hx)?);
        }

        let mut ops_fwd = Vec::<HistoryOp>::new();
        let mut qid_to_block = Vec::<u64>::new();
        let mut qpos = 0usize;

        let t_journal = t_start();
        for_each_line_filtered(
            journal,
            &self.srs_id,
            self.n,
            self.logn,
            |j: JournalLine| -> Result<()> {
                let b = j.block_number;
                if b <= start_block || b > self.last_block {
                    return Ok(());
                }

                let mut delta = Vec::<Fr>::new();
                for hx in &j.delta_hex {
                    delta.push(fr_from_hex(hx)?);
                }

                let (beta, delta) =
                    canonicalize_beta_delta(&j.changed_indices, &delta);

                if !beta.is_empty() {
                    ops_fwd.push(HistoryOp::Update { beta, delta });
                }

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
        let dt_journal = t_us(t_journal);
        eprintln!(
            "[ProofServerState::history_query_same_alpha] journal blocks={} queries={} micros={}",
            blocks.len(),
            qid_to_block.len(),
            dt_journal
        );

        let mut ops_rev = ops_fwd;
        ops_rev.reverse();
        for op in ops_rev.iter_mut() {
            if let HistoryOp::Update { delta, .. } = op {
                for d in delta.iter_mut() {
                    *d = -*d;
                }
            }
        }

        let t_core = t_start();
        let results = vupdate_history_same_alpha(
            ctx,
            &self.alpha_indices,
            &gq_final,
            &ops_rev,
        );
        let dt_core = t_us(t_core);
        eprintln!(
            "[ProofServerState::history_query_same_alpha] rewind queries={} micros={}",
            qid_to_block.len(),
            dt_core
        );

        let mut out = Vec::new();
        for r in results {
            out.push((qid_to_block[r.query_id], r.proofs));
        }

        out.sort_by_key(|(b, _)| *b);
        let dt_total = t_us(t_total);
        eprintln!(
            "[ProofServerState::history_query_same_alpha] total queries={} micros={}",
            qid_to_block.len(),
            dt_total
        );
        Ok(out)
    }
}
