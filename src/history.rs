use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::{Zero};

use crate::vc_context::VcContext;

#[derive(Clone, Debug)]
pub enum HistoryOp {
    Update { beta: Vec<usize>, delta: Vec<Fr> },
    Query { query_id: usize },
}

/// Result for a single history query: indices α and their witnesses.
#[derive(Clone, Debug)]
pub struct HistoryQueryResult {
    pub query_id: usize,
    pub indices: Vec<usize>, // α
    pub proofs: Vec<G1>,     // witnesses for α, same order
}

fn merge_beta_delta(
    beta_a: &[usize],
    delta_a: &[Fr],
    beta_b: &[usize],
    delta_b: &[Fr],
) -> (Vec<usize>, Vec<Fr>) {
    debug_assert_eq!(beta_a.len(), delta_a.len());
    debug_assert_eq!(beta_b.len(), delta_b.len());

    let mut out_b = Vec::with_capacity(beta_a.len() + beta_b.len());
    let mut out_d = Vec::with_capacity(beta_a.len() + beta_b.len());

    let mut i = 0usize;
    let mut j = 0usize;
    while i < beta_a.len() && j < beta_b.len() {
        let ba = beta_a[i];
        let bb = beta_b[j];
        if ba == bb {
            let d = delta_a[i] + delta_b[j];
            if !d.is_zero() {
                out_b.push(ba);
                out_d.push(d);
            }
            i += 1;
            j += 1;
        } else if ba < bb {
            if !delta_a[i].is_zero() {
                out_b.push(ba);
                out_d.push(delta_a[i]);
            }
            i += 1;
        } else {
            if !delta_b[j].is_zero() {
                out_b.push(bb);
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

/// Canonicalize a (beta,delta) by sorting and summing duplicates.
/// This is used at UPDATE leaves (typically tiny per-block).
fn canonicalize_beta_delta(beta: &[usize], delta: &[Fr]) -> (Vec<usize>, Vec<Fr>) {
    debug_assert_eq!(beta.len(), delta.len());

    // beta sizes per block are small; O(k log k) here is fine
    let mut pairs: Vec<(usize, Fr)> = beta.iter().copied().zip(delta.iter().copied()).collect();
    pairs.sort_by_key(|(b, _)| *b);

    let mut out_b: Vec<usize> = Vec::with_capacity(pairs.len());
    let mut out_d: Vec<Fr> = Vec::with_capacity(pairs.len());

    let mut i = 0usize;
    while i < pairs.len() {
        let b = pairs[i].0;
        let mut acc = pairs[i].1;
        i += 1;
        while i < pairs.len() && pairs[i].0 == b {
            acc += pairs[i].1;
            i += 1;
        }
        if !acc.is_zero() {
            out_b.push(b);
            out_d.push(acc);
        }
    }

    (out_b, out_d)
}

#[derive(Clone, Debug)]
struct Node {
    l: usize,
    r: usize, // [l, r)
    has_query: bool,

    // merged updates (β,Δ) for updates in [l,r), sorted by β, unique
    beta: Vec<usize>,
    delta: Vec<Fr>,

    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

fn build_node(stream: &[HistoryOp], l: usize, r: usize) -> Node {
    debug_assert!(l < r);

    if r - l == 1 {
        match &stream[l] {
            HistoryOp::Query { .. } => Node {
                l,
                r,
                has_query: true,
                beta: Vec::new(),
                delta: Vec::new(),
                left: None,
                right: None,
            },
            HistoryOp::Update { beta, delta } => {
                let (b, d) = canonicalize_beta_delta(beta, delta);
                Node {
                    l,
                    r,
                    has_query: false,
                    beta: b,
                    delta: d,
                    left: None,
                    right: None,
                }
            }
        }
    } else {
        let m = l + (r - l) / 2;
        let left = build_node(stream, l, m);
        let right = build_node(stream, m, r);

        let has_query = left.has_query || right.has_query;
        let (beta, delta) = merge_beta_delta(&left.beta, &left.delta, &right.beta, &right.delta);

        Node {
            l,
            r,
            has_query,
            beta,
            delta,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }
}

/// Paper-faithful recursion:
/// - recurse left with current witnesses
/// - update witnesses by left updates
/// - recurse right with updated witnesses
fn vupdate_run(
    ctx: &VcContext,
    stream: &[HistoryOp],
    node: &Node,
    alpha_all: &[usize],
    gq_live: &[G1],
    out: &mut Vec<HistoryQueryResult>,
) {
    if !node.has_query {
        return;
    }

    // leaf
    if node.r - node.l == 1 {
        if let HistoryOp::Query { query_id } = &stream[node.l] {
            out.push(HistoryQueryResult {
                query_id: *query_id,
                indices: alpha_all.to_vec(),
                proofs: gq_live.to_vec(),
            });
        }
        return;
    }

    let left = node.left.as_ref().expect("internal node missing left");
    let right = node.right.as_ref().expect("internal node missing right");

    // Left recurse (no witness changes before left)
    if left.has_query {
        vupdate_run(ctx, stream, left, alpha_all, gq_live, out);
    }

    // Right recurse: apply left updates to witnesses, then recurse right
    if right.has_query {
        let mut gq_r = gq_live.to_vec();

        if !left.beta.is_empty() && !alpha_all.is_empty() {
            // Optional fallback: pairwise (correct but can be slower)
            let use_pairwise = std::env::var_os("CAUCHY_HISTORY_PAIRWISE").is_some();
            if use_pairwise {
                for (ai, &aidx) in alpha_all.iter().enumerate() {
                    let mut w = gq_r[ai];
                    for (&bidx, &d) in left.beta.iter().zip(left.delta.iter()) {
                        if !d.is_zero() {
                            w = ctx.update_witness(aidx, w, bidx, d);
                        }
                    }
                    gq_r[ai] = w;
                }
            } else {
                gq_r = ctx.update_witnesses_batch(alpha_all, &gq_r, &left.beta, &left.delta);
            }
        }

        vupdate_run(ctx, stream, right, alpha_all, &gq_r, out);
    }
}

pub fn vupdate_history_same_alpha(
    ctx: &VcContext,
    alpha_all: &[usize],
    gq_alpha_final: &[G1],
    stream_reversed: &[HistoryOp],
) -> Vec<HistoryQueryResult> {
    if stream_reversed.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(alpha_all.len(), gq_alpha_final.len());

    let root = build_node(stream_reversed, 0, stream_reversed.len());
    if !root.has_query {
        return Vec::new();
    }

    let mut out = Vec::new();
    vupdate_run(
        ctx,
        stream_reversed,
        &root,
        alpha_all,
        gq_alpha_final,
        &mut out,
    );
    out
}
