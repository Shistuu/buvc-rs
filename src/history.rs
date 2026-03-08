// src/history.rs
use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::{Field, Zero};

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::vc_context::VcContext;
use crate::types::{HistoryOp, HistoryQueryResult};

#[derive(Clone)]
struct QueryItem {
    pos: usize,            // position in stream_rev
    query_id: usize,       // your block number casted
    proofs: Arc<Vec<G1>>,  // witnesses for alpha at that query
}

/// Apply a list of updates (referenced by their positions in stream_rev) to `base` proofs.
/// We avoid giant MSMs by chunking: accumulate into a bounded map, flush into
/// update_witnesses_batch when the map gets big.
fn apply_updates_by_positions_chunked(
    ctx: &VcContext,
    alpha: &[usize],
    base: &[G1],
    stream_rev: &[HistoryOp],
    update_positions: &[usize],
) -> Vec<G1> {
    if alpha.is_empty() || update_positions.is_empty() {
        return base.to_vec();
    }

    // Tuning knobs: keep each batch update bounded.
    // If this is too small, you do too many calls; if too large, you risk huge MSM again.
    const MAX_BETA_PER_CALL: usize = 2048;
    const MAX_UPDATES_PER_FLUSH: usize = 32;


    let mut cur = base.to_vec();

    let mut acc: BTreeMap<usize, Fr> = BTreeMap::new();
    let mut updates_since_flush: usize = 0;

    let flush = |cur: &mut Vec<G1>, acc: &mut BTreeMap<usize, Fr>| {
        if acc.is_empty() {
            return;
        }

        // BTreeMap is already sorted by key => canonical beta order.
        let mut beta = Vec::with_capacity(acc.len());
        let mut delta = Vec::with_capacity(acc.len());
        for (&i, &d) in acc.iter() {
            if !d.is_zero() {
                beta.push(i);
                delta.push(d);
            }
        }
        acc.clear();

        if beta.is_empty() {
            return;
        }

        *cur = ctx.update_witnesses_batch(alpha, cur, &beta, &delta);
    };

    for &p in update_positions {
        match &stream_rev[p] {
            HistoryOp::Update { beta, delta } => {
                #[cfg(debug_assertions)]
                debug_assert_canonical_update(beta, delta);

                // Accumulate this update into acc
                for (&i, &d) in beta.iter().zip(delta.iter()) {
                    if d.is_zero() {
                        continue;
                    }
                    let e = acc.entry(i).or_insert(Fr::ZERO);
                    *e += d;
                    if e.is_zero() {
                        acc.remove(&i);
                    }
                }
                updates_since_flush += 1;

                // Flush if acc got too big or we've buffered enough updates.
                if acc.len() >= MAX_BETA_PER_CALL || updates_since_flush >= MAX_UPDATES_PER_FLUSH {
                    flush(&mut cur, &mut acc);
                    updates_since_flush = 0;
                }
            }
            HistoryOp::Query { .. } => {
                // Queries contribute no updates
            }
        }
    }

    // Final flush
    flush(&mut cur, &mut acc);
    cur
}

/// Build a segment tree where each node stores a list of stream positions that are Update ops
/// in that interval [lo, hi). The list is in increasing position order, preserving stream order.
fn build_update_pos_tree_rec(
    node: usize,
    lo: usize,
    hi: usize,
    stream: &[HistoryOp],
    tree: &mut [Vec<usize>],
) {
    if hi - lo == 1 {
        tree[node] = match &stream[lo] {
            HistoryOp::Update { .. } => {
                #[cfg(debug_assertions)]
                debug_assert_canonical_update(beta, delta);
                vec![lo]
            }
            HistoryOp::Query { .. } => Vec::new(),
        };
        return;
    }

    let m = lo + (hi - lo) / 2;
    build_update_pos_tree_rec(node * 2, lo, m, stream, tree);
    build_update_pos_tree_rec(node * 2 + 1, m, hi, stream, tree);

    // Concatenate left then right (preserves order)
    let left = tree[node * 2].clone();
    let right = tree[node * 2 + 1].clone();

    let mut out = Vec::with_capacity(left.len() + right.len());
    out.extend_from_slice(&left);
    out.extend_from_slice(&right);
    tree[node] = out;
}

/// Paper-style VUpdate(S, α⃗) with divide-and-conquer,
/// but summaries are lists of per-block updates (no giant merged beta/delta).
pub fn vupdate_history_vupdate_dc(
    ctx: &VcContext,
    alpha: &[usize],
    gq_head: &[G1],
    stream_rev: &[HistoryOp],
) -> Vec<HistoryQueryResult> {
    if stream_rev.is_empty() {
        return Vec::new();
    }

    // Collect queries; each starts from head proofs
    let head_arc = Arc::new(gq_head.to_vec());
    let mut queries: Vec<QueryItem> = Vec::new();

    for (pos, op) in stream_rev.iter().enumerate() {
        match op {
            HistoryOp::Query { query_id } => {
                queries.push(QueryItem {
                    pos,
                    query_id: *query_id,
                    proofs: head_arc.clone(),
                });
            }
            HistoryOp::Update { .. } => {
                #[cfg(debug_assertions)]
                debug_assert_canonical_update(beta, delta);
            }
        }
    }

    if queries.is_empty() {
        return Vec::new();
    }

    // Build update-position segment tree once
    let n = stream_rev.len();
    let mut tree: Vec<Vec<usize>> = vec![Vec::new(); 4 * n + 5];
    build_update_pos_tree_rec(1, 0, n, stream_rev, &mut tree);

    // Recurse
    vupdate_rec_positions(ctx, alpha, stream_rev, &tree, 1, 0, n, &mut queries);

    // Emit results
    queries
        .into_iter()
        .map(|q| HistoryQueryResult {
            query_id: q.query_id,
            indices: alpha.to_vec(),
            proofs: (*q.proofs).clone(),
        })
        .collect()
}

fn vupdate_rec_positions(
    ctx: &VcContext,
    alpha: &[usize],
    stream_rev: &[HistoryOp],
    tree: &[Vec<usize>],
    node: usize,
    lo: usize,
    hi: usize,
    queries: &mut [QueryItem],
) {
    if queries.is_empty() || hi - lo <= 1 {
        return;
    }

    let m = lo + (hi - lo) / 2;

    // Split queries by position relative to m
    let mut split = 0usize;
    while split < queries.len() && queries[split].pos < m {
        split += 1;
    }
    let (ql, qr) = queries.split_at_mut(split);

    // Step (paper-style): update right queries using updates from left interval [lo, m)
    if !qr.is_empty() && !alpha.is_empty() {
        let left_updates = &tree[node * 2];
        if !left_updates.is_empty() {
            // All queries in qr should share the same base Arc in this subtree.
            #[cfg(debug_assertions)]
            {
                let p0 = Arc::as_ptr(&qr[0].proofs);
                for q in qr.iter() {
                    debug_assert!(
                        Arc::as_ptr(&q.proofs) == p0,
                        "proofs diverged inside subtree (unexpected)"
                    );
                }
            }

            let base_arc = qr[0].proofs.clone();
            let updated = apply_updates_by_positions_chunked(
                ctx,
                alpha,
                base_arc.as_slice(),
                stream_rev,
                left_updates,
            );

            let updated_arc = Arc::new(updated);
            for q in qr.iter_mut() {
                q.proofs = updated_arc.clone();
            }
        }
    }

    vupdate_rec_positions(ctx, alpha, stream_rev, tree, node * 2, lo, m, ql);
    vupdate_rec_positions(ctx, alpha, stream_rev, tree, node * 2 + 1, m, hi, qr);
}
