// src/history.rs
use std::collections::{BTreeMap, BTreeSet};

use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::{Field, Zero};

use crate::vc_context::VcContext;

/// One element of the (already reversed + negated) history stream S, as in §2.5:
///   - UPDATE(β_i, Δv_{β_i})
///   - QUERY(α_j)
///
/// NOTE: This is now the *general* form: each query carries its own α-set.
/// The caller is responsible for ensuring α_j ⊆ α_all (the union) and
/// that the final witnesses they pass in correspond to α_all at the final state.
#[derive(Clone, Debug)]
pub enum HistoryOp {
    /// UPDATE(β, Δ) in the *reversed* stream.
    /// Precondition: deltas are already negated compared to forward time.
    Update {
        beta: Vec<usize>, // indices in [0..N)
        delta: Vec<Fr>,   // aligned with beta
    },

    /// QUERY at some logical point in the stream.
    /// query_id is an opaque handle; α is the indices whose proofs we want.
    Query {
        query_id: usize,
        alpha: Vec<usize>, // indices whose proofs are requested for this query
    },
}

/// Result for a single history query: indices α_j and their witnesses.
#[derive(Clone, Debug)]
pub struct HistoryQueryResult {
    pub query_id: usize,
    pub indices: Vec<usize>, // α_j
    pub proofs: Vec<G1>,     // witnesses for α_j, same order as indices
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

/// Internal recursive VUpdate(S, α⃗) as in the paper (Section 2.5 / Section 7).
///
/// - `alpha_live` are the indices whose proofs we currently maintain at this
///   node in the recursion (this is α⃗ for this substream).
/// - `gq_live` are the witnesses for `alpha_live`, same order.
/// - `stream` is the (reversed+negated) subsequence S for this node.
///
/// Invariant: for every QUERY in `stream`, its α_j ⊆ alpha_live.
fn vupdate_inner(
    ctx: &VcContext,
    alpha_live: &[usize],
    gq_live: &[G1],
    stream: &[HistoryOp],
    out: &mut Vec<HistoryQueryResult>,
) {
    if stream.is_empty() {
        return;
    }

    // Base case: |S| = 1
    if stream.len() == 1 {
        if let HistoryOp::Query { query_id, alpha } = &stream[0] {
            if alpha.is_empty() {
                // nothing requested; still return a result with empty proofs
                out.push(HistoryQueryResult {
                    query_id: *query_id,
                    indices: Vec::new(),
                    proofs: Vec::new(),
                });
                return;
            }

            // Map: index -> position in alpha_live
            let mut pos: BTreeMap<usize, usize> = BTreeMap::new();
            for (p, &idx) in alpha_live.iter().enumerate() {
                pos.insert(idx, p);
            }

            let mut proofs = Vec::<G1>::with_capacity(alpha.len());
            for &i in alpha {
                let p = pos
                    .get(&i)
                    .expect("query alpha index not in alpha_live (invariant broken)");
                proofs.push(gq_live[*p]);
            }

            out.push(HistoryQueryResult {
                query_id: *query_id,
                indices: alpha.clone(),
                proofs,
            });
        }
        // If it's an UPDATE-only singleton, nothing to output.
        return;
    }

    // Split S into S_l and S_r at m = floor(|S| / 2).
    let m = stream.len() / 2;
    let (s_l, s_r) = stream.split_at(m);

    // Compute α_l and α_r as unions of α_j for queries in S_l / S_r.
    let mut alpha_l_set: BTreeSet<usize> = BTreeSet::new();
    let mut alpha_r_set: BTreeSet<usize> = BTreeSet::new();
    let mut has_query_l = false;
    let mut has_query_r = false;

    for op in s_l {
        if let HistoryOp::Query { alpha, .. } = op {
            has_query_l = true;
            for &i in alpha {
                alpha_l_set.insert(i);
            }
        }
    }
    for op in s_r {
        if let HistoryOp::Query { alpha, .. } = op {
            has_query_r = true;
            for &i in alpha {
                alpha_r_set.insert(i);
            }
        }
    }

    // If neither half has any queries, we're done.
    if !has_query_l && !has_query_r {
        return;
    }

    // Map from index -> position in current alpha_live, reused for both halves.
    let mut live_pos: BTreeMap<usize, usize> = BTreeMap::new();
    for (p, &idx) in alpha_live.iter().enumerate() {
        live_pos.insert(idx, p);
    }

    // Gather left-half updates (β_l, Δ_l) and canonicalize.
    let mut beta_l = Vec::<usize>::new();
    let mut delta_l = Vec::<Fr>::new();
    for op in s_l {
        if let HistoryOp::Update { beta, delta } = op {
            beta_l.extend_from_slice(beta);
            delta_l.extend_from_slice(delta);
        }
    }
    let (beta_l, delta_l) = canonicalize_beta_delta(&beta_l, &delta_l);

    // Left side: recurse with α_l and its witnesses (projection of alpha_live/gq_live).
    if has_query_l && !alpha_l_set.is_empty() {
        let alpha_l: Vec<usize> = alpha_l_set.iter().copied().collect();
        let mut gq_l = Vec::<G1>::with_capacity(alpha_l.len());
        for &idx in &alpha_l {
            let p = live_pos
                .get(&idx)
                .expect("alpha_l index not in alpha_live (invariant broken)");
            gq_l.push(gq_live[*p]);
        }
        vupdate_inner(ctx, &alpha_l, &gq_l, s_l, out);
    }

    // Right side: recurse with α_r, after applying all S_l updates to its proofs.
    if has_query_r && !alpha_r_set.is_empty() {
        let alpha_r: Vec<usize> = alpha_r_set.iter().copied().collect();
        let mut gq_r = Vec::<G1>::with_capacity(alpha_r.len());
        for &idx in &alpha_r {
            let p = live_pos
                .get(&idx)
                .expect("alpha_r index not in alpha_live (invariant broken)");
            gq_r.push(gq_live[*p]);
        }

        if !beta_l.is_empty() {
            // Batch update proofs for α_r with the accumulated left-half updates.
            gq_r = ctx.update_witnesses_batch(&alpha_r, &gq_r, &beta_l, &delta_l);
        }

        vupdate_inner(ctx, &alpha_r, &gq_r, s_r, out);
    }
}

/// Public entry point: VUpdate(S, α⃗) in the general form.
///
/// - `ctx` is your VcContext.
/// - `alpha_all` are the indices for which you have final witnesses at the
///   time corresponding to the *start* of `stream_reversed` (i.e., the
///   final state in forward time).
/// - `gq_alpha_all_final` are those witnesses, same order as `alpha_all`.
/// - `stream_reversed` is the *reversed+negated* stream S as in §2.5.
///
/// For each QUERY in `stream_reversed` with a requested α_j, this returns:
///   HistoryQueryResult { query_id, indices = α_j, proofs = witnesses at that point }.
///
/// Invariant required from the caller:
///   For every QUERY{alpha = α_j}, we must have α_j ⊆ α_all.
pub fn vupdate_history(
    ctx: &VcContext,
    alpha_all: &[usize],
    gq_alpha_all_final: &[G1],
    stream_reversed: &[HistoryOp],
) -> Vec<HistoryQueryResult> {
    // Collect the union of all α_j across the whole stream.
    let mut alpha_union_set: BTreeSet<usize> = BTreeSet::new();
    let mut any_query = false;
    for op in stream_reversed {
        if let HistoryOp::Query { alpha, .. } = op {
            any_query = true;
            for &i in alpha {
                alpha_union_set.insert(i);
            }
        }
    }

    if !any_query {
        return Vec::new();
    }
    if alpha_union_set.is_empty() {
        // There are queries, but all α_j are empty; they all yield empty proofs.
        // We can handle this as a special case without recursing.
        let mut out = Vec::new();
        for op in stream_reversed {
            if let HistoryOp::Query { query_id, alpha } = op {
                debug_assert!(alpha.is_empty());
                out.push(HistoryQueryResult {
                    query_id: *query_id,
                    indices: Vec::new(),
                    proofs: Vec::new(),
                });
            }
        }
        return out;
    }

    // alpha_union is the "α⃗" for the full stream: union of all per-query α_j.
    let alpha_union: Vec<usize> = alpha_union_set.iter().copied().collect();

    // Map alpha_all -> positions, then restrict to alpha_union to build the initial gq_live.
    let mut pos_all: BTreeMap<usize, usize> = BTreeMap::new();
    for (p, &idx) in alpha_all.iter().enumerate() {
        pos_all.insert(idx, p);
    }

    let mut gq_union = Vec::<G1>::with_capacity(alpha_union.len());
    for &idx in &alpha_union {
        let p = pos_all
            .get(&idx)
            .expect("alpha_union index not in alpha_all (caller invariant broken)");
        gq_union.push(gq_alpha_all_final[*p]);
    }

    let mut outputs = Vec::new();
    vupdate_inner(ctx, &alpha_union, &gq_union, stream_reversed, &mut outputs);
    outputs
}
