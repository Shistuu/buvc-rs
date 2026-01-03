// src/proof_server.rs
use std::collections::{BTreeMap, HashMap};

use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::Zero;
use eyre::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::codec::{g1_from_hex, g1_to_hex};
use crate::types::UserState;
use crate::vc_context::VcContext;

/// A single logical user of the proof server, identified however you like.
/// We store the indices this user cares about (same meaning as in UserState).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProofServerUser {
    pub user_id: String,
    pub alpha_indices: Vec<usize>,
}

/// State of a proof-serving node that maintains proofs for the *union* of
/// all α-sets across many users.
///
/// This is the "multi-user" object the paper talks about: a single
/// vector of proofs that is updated once per block and can be used
/// to answer proofs for any registered user.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProofServerState {
    pub n: usize,
    pub logn: usize,
    pub srs_id: String,

    pub last_block: u64,
    pub gc_hex: String,

    /// Union of all user indices (sorted, unique).
    pub alpha_indices: Vec<usize>,
    /// Witnesses for `alpha_indices`, same order.
    pub alpha_witnesses_hex: Vec<String>,

    /// Logical users and which indices they care about.
    pub users: Vec<ProofServerUser>,

    pub universe_mode: String,
    pub witness_mode: String, // e.g. "user_maintains_witnesses_vupdate"
}

impl ProofServerState {
    /// Construct a proof-server state from a collection of (user_id, UserState).
    ///
    /// All user states must:
    ///   - Share the same n, logn, srs_id, last_block, gc_hex,
    ///     universe_mode, witness_mode.
    ///   - Have consistent witnesses for overlapping indices.
    pub fn from_user_states(users: &[(String, UserState)]) -> Result<Self> {
        if users.is_empty() {
            bail!("need at least one user to build ProofServerState");
        }

        let (_, first) = &users[0];
        let n = first.n;
        let logn = first.logn;
        let srs_id = first.srs_id.clone();
        let last_block = first.last_block;
        let gc_hex = first.gc_hex.clone();
        let universe_mode = first.universe_mode.clone();
        let witness_mode = first.witness_mode.clone();

        // Sanity-check invariants across all users.
        for (_, u) in users.iter().skip(1) {
            if u.n != n
                || u.logn != logn
                || u.srs_id != srs_id
                || u.last_block != last_block
                || u.gc_hex != gc_hex
                || u.universe_mode != universe_mode
                || u.witness_mode != witness_mode
            {
                bail!("inconsistent UserState; cannot merge into a single proof server");
            }
        }

        // Build union of indices with canonical witnesses.
        // For each index i, we remember one witness and check all others match.
        let mut index_to_wit: BTreeMap<usize, G1> = BTreeMap::new();

        for (_, u) in users {
            for (idx, wit_hex) in u.alpha_indices.iter().zip(u.alpha_witnesses_hex.iter()) {
                let g = g1_from_hex(wit_hex)?;
                match index_to_wit.get(idx) {
                    Some(existing) => {
                        if existing != &g {
                            bail!("inconsistent witness for index {} across users", idx);
                        }
                    }
                    None => {
                        index_to_wit.insert(*idx, g);
                    }
                }
            }
        }

        let alpha_indices: Vec<usize> = index_to_wit.keys().copied().collect();
        let alpha_witnesses_hex: Vec<String> =
            index_to_wit.values().map(|g| g1_to_hex(g)).collect();

        let users_desc = users
            .iter()
            .map(|(id, u)| ProofServerUser {
                user_id: id.clone(),
                alpha_indices: u.alpha_indices.clone(),
            })
            .collect();

        Ok(Self {
            n,
            logn,
            srs_id,
            last_block,
            gc_hex,
            alpha_indices,
            alpha_witnesses_hex,
            users: users_desc,
            universe_mode,
            witness_mode,
        })
    }

    /// Apply a single block's updates (β, Δ) to *all* users at once.
    ///
    /// This is the multi-user VUpdate: the proof server only calls
    /// update_witnesses_batch once on the union of all α, then each
    /// user can be served from the updated vector.
    pub fn apply_update(
        &mut self,
        ctx: &VcContext,
        beta: &[usize],
        delta: &[Fr],
        new_block: u64,
        new_gc_hex: &str,
    ) -> Result<()> {
        if beta.len() != delta.len() {
            bail!("beta/delta length mismatch");
        }
        if new_block <= self.last_block {
            bail!(
                "new_block {} must be > last_block {}",
                new_block,
                self.last_block
            );
        }

        // Decode witnesses to G1.
        let mut gq: Vec<G1> = Vec::with_capacity(self.alpha_witnesses_hex.len());
        for hx in &self.alpha_witnesses_hex {
            gq.push(g1_from_hex(hx)?);
        }

        // Update commitment and witnesses, mirroring your vupdate_step.
        let mut gc = g1_from_hex(&self.gc_hex)?;
        for (&i, &d) in beta.iter().zip(delta.iter()) {
            if !d.is_zero() {
                gc = ctx.update_commitment(gc, i, d);
            }
        }

        // Batch update all maintained proofs.
        let gq_new = ctx.update_witnesses_batch(&self.alpha_indices, &gq, beta, delta);

        // Check against publisher-pinned commitment, if provided.
        if !new_gc_hex.is_empty() {
            let pinned = new_gc_hex.trim();
            if pinned != g1_to_hex(&gc) {
                bail!("commitment mismatch in proof server at block {}", new_block);
            }
        }

        // Persist.
        self.last_block = new_block;
        self.gc_hex = g1_to_hex(&gc);
        self.alpha_witnesses_hex = gq_new.iter().map(g1_to_hex).collect();

        Ok(())
    }

    /// Serve one user's current witnesses as (indices, witnesses_hex).
    ///
    /// The caller can then pair these with values to form SingleProofs
    /// or aggregated proofs, exactly like your existing UserQuery.
    pub fn serve_user(&self, user_id: &str) -> Option<(Vec<usize>, Vec<String>)> {
        let user = self.users.iter().find(|u| u.user_id == user_id)?;
        // Build a map from global index -> position in alpha_indices.
        let pos: HashMap<usize, usize> = self
            .alpha_indices
            .iter()
            .enumerate()
            .map(|(i, idx)| (*idx, i))
            .collect();

        let mut wits = Vec::with_capacity(user.alpha_indices.len());
        for idx in &user.alpha_indices {
            let p = *pos.get(idx)?;
            wits.push(self.alpha_witnesses_hex[p].clone());
        }

        Some((user.alpha_indices.clone(), wits))
    }
}
