use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::Zero;

use crate::{
    vc_context::VcContext,
    types::{HistoryOp, HistoryQueryResult},
};

/// Merge two sorted (beta, delta) lists
fn merge_beta_delta(
    beta_a: &[usize],
    delta_a: &[Fr],
    beta_b: &[usize],
    delta_b: &[Fr],
) -> (Vec<usize>, Vec<Fr>) {
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
            if !delta_a[i].is_zero() {
                out_b.push(beta_a[i]);
                out_d.push(delta_a[i]);
            }
            i += 1;
        } else {
            if !delta_b[j].is_zero() {
                out_b.push(beta_b[j]);
                out_d.push(delta_b[j]);
            }
            j += 1;
        }
    }

    while i < beta_a.len() {
        if !delta_a[i].is_zero() {
            out_b.push(beta_a[i]);
            out_d.push(delta_a[i]);
        }
        i += 1;
    }

    while j < beta_b.len() {
        if !delta_b[j].is_zero() {
            out_b.push(beta_b[j]);
            out_d.push(delta_b[j]);
        }
        j += 1;
    }

    (out_b, out_d)
}

/// Apply history updates lazily (paper-faithful)
pub fn vupdate_history_same_alpha(
    ctx: &VcContext,
    alpha: &[usize],
    gq_head: &[G1],
    stream_rev: &[HistoryOp],
) -> Vec<HistoryQueryResult> {
    let mut out = Vec::new();
    let mut gq_live = gq_head.to_vec();

    let mut acc_beta: Vec<usize> = Vec::new();
    let mut acc_delta: Vec<Fr> = Vec::new();

    for op in stream_rev {
        match op {
            HistoryOp::Update { beta, delta } => {
                let (b, d) = merge_beta_delta(
                    &acc_beta,
                    &acc_delta,
                    beta,
                    delta,
                );
                acc_beta = b;
                acc_delta = d;
            }

            HistoryOp::Query { query_id } => {
                if !acc_beta.is_empty() {
                    gq_live = ctx.update_witnesses_batch(
                        alpha,
                        &gq_live,
                        &acc_beta,
                        &acc_delta,
                    );
                    acc_beta.clear();
                    acc_delta.clear();
                }

                out.push(HistoryQueryResult {
                    query_id: *query_id,
                    indices: alpha.to_vec(),
                    proofs: gq_live.clone(),
                });
            }
        }
    }

    out
}
