use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::Zero;
use std::sync::Arc;

use crate::{
    vc_context::VcContext,
    types::{HistoryOp, HistoryQueryResult},
};

#[derive(Clone)]
struct QueryItem {
    pos: usize,
    query_id: usize,
    proofs: Arc<Vec<G1>>,
}

#[derive(Clone, Default)]
struct UpdateSummary {
    beta: Vec<usize>, // strictly increasing
    delta: Vec<Fr>,   // non-zero entries only (preferred)
}

#[inline]
fn debug_assert_sorted_dedup(beta: &[usize], delta: &[Fr]) {
    debug_assert_eq!(beta.len(), delta.len(), "beta/delta length mismatch");
    for w in beta.windows(2) {
        debug_assert!(w[0] < w[1], "beta not strictly increasing");
    }
    // Stronger invariant (recommended):
    for &d in delta {
        debug_assert!(!d.is_zero(), "delta contains zero (not canonical)");
    }
}

/// Merge two sorted+deduped lists, summing overlaps, dropping zeros.
fn merge_beta_delta(
    beta_a: &[usize],
    delta_a: &[Fr],
    beta_b: &[usize],
    delta_b: &[Fr],
) -> UpdateSummary {
    debug_assert_sorted_dedup(beta_a, delta_a);
    debug_assert_sorted_dedup(beta_b, delta_b);

    let mut out_b = Vec::with_capacity(beta_a.len() + beta_b.len());
    let mut out_d = Vec::with_capacity(beta_a.len() + beta_b.len());

    let mut i = 0usize;
    let mut j = 0usize;

    while i < beta_a.len() && j < beta_b.len() {
        if beta_a[i] == beta_b[j] {
            let d = delta_a[i] + delta_b[j];
            if !d.is_zero() {
                out_b.push(beta_a[i]);
                out_d.push(d);
            }
            i += 1;
            j += 1;
        } else if beta_a[i] < beta_b[j] {
            out_b.push(beta_a[i]);
            out_d.push(delta_a[i]);
            i += 1;
        } else {
            out_b.push(beta_b[j]);
            out_d.push(delta_b[j]);
            j += 1;
        }
    }

    while i < beta_a.len() {
        out_b.push(beta_a[i]);
        out_d.push(delta_a[i]);
        i += 1;
    }
    while j < beta_b.len() {
        out_b.push(beta_b[j]);
        out_d.push(delta_b[j]);
        j += 1;
    }

    UpdateSummary { beta: out_b, delta: out_d }
}

/// Build recursion-aligned summaries: sums[node] is updates for interval [lo, hi).
fn build_summaries_rec(
    node: usize,
    lo: usize,
    hi: usize,
    stream: &[HistoryOp],
    sums: &mut [UpdateSummary],
) {
    if hi - lo == 1 {
        match &stream[lo] {
            HistoryOp::Update { beta, delta } => {
                debug_assert_sorted_dedup(beta, delta);
                sums[node] = UpdateSummary { beta: beta.clone(), delta: delta.clone() };
            }
            HistoryOp::Query { .. } => {
                sums[node] = UpdateSummary::default();
            }
        }
        return;
    }

    let m = lo + (hi - lo) / 2;
    build_summaries_rec(node * 2, lo, m, stream, sums);
    build_summaries_rec(node * 2 + 1, m, hi, stream, sums);

    let left = &sums[node * 2];
    let right = &sums[node * 2 + 1];
    sums[node] = merge_beta_delta(&left.beta, &left.delta, &right.beta, &right.delta);
}

/// Paper-faithful divide-and-conquer VUpdate(S, α⃗).
pub fn vupdate_history_vupdate_dc_fast(
    ctx: &VcContext,
    alpha: &[usize],
    gq_head: &[G1],
    stream_rev: &[HistoryOp],
) -> Vec<HistoryQueryResult> {
    if stream_rev.is_empty() {
        return Vec::new();
    }

    let head_arc = Arc::new(gq_head.to_vec());
    let mut queries: Vec<QueryItem> = Vec::new();

    for (pos, op) in stream_rev.iter().enumerate() {
        match op {
            HistoryOp::Query { query_id } => queries.push(QueryItem {
                pos,
                query_id: *query_id,
                proofs: head_arc.clone(),
            }),
            HistoryOp::Update { beta, delta } => debug_assert_sorted_dedup(beta, delta),
        }
    }
    if queries.is_empty() {
        return Vec::new();
    }

    let n = stream_rev.len().max(1);
    let mut sums = vec![UpdateSummary::default(); 4 * n + 5];
    build_summaries_rec(1, 0, stream_rev.len(), stream_rev, &mut sums);

    vupdate_rec_fast(ctx, alpha, &sums, 1, 0, stream_rev.len(), &mut queries);

    queries
        .into_iter()
        .map(|q| HistoryQueryResult {
            query_id: q.query_id,
            indices: alpha.to_vec(),
            proofs: (*q.proofs).clone(),
        })
        .collect()
}

fn vupdate_rec_fast(
    ctx: &VcContext,
    alpha: &[usize],
    sums: &[UpdateSummary],
    node: usize,
    lo: usize,
    hi: usize,
    queries: &mut [QueryItem],
) {
    if queries.is_empty() || hi - lo == 1 {
        return; // Step 1
    }

    let m = lo + (hi - lo) / 2; // Step 2

    let mut split = 0usize;
    while split < queries.len() && queries[split].pos < m {
        split += 1;
    }
    let (ql, qr) = queries.split_at_mut(split);

    // Step 3+4: left interval updates affect all right queries.
    if !qr.is_empty() {
        let left_sum = &sums[node * 2];
        if !left_sum.beta.is_empty() {
            let base = &qr[0].proofs;
            let updated_vec = ctx.update_witnesses_batch(
                alpha,
                base.as_slice(),
                &left_sum.beta,
                &left_sum.delta,
            );
            let updated_arc = Arc::new(updated_vec);
            for q in qr.iter_mut() {
                q.proofs = updated_arc.clone();
            }
        }
    }

    // Step 5
    vupdate_rec_fast(ctx, alpha, sums, node * 2, lo, m, ql);
    vupdate_rec_fast(ctx, alpha, sums, node * 2 + 1, m, hi, qr);
}
