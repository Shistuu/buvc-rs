// src/types.rs
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SnapshotState {
    pub block_number: u64,
    /// address_hex -> balance_wei decimal string
    pub balances: std::collections::HashMap<String, String>,
    /// encoding identifier, e.g. "u256"
    pub balance_encoding: String,
}

/// Publisher snapshot: public commitment at a block.
/// Paper-faithful: snapshot stores ONLY the commitment C and metadata.
/// No witnesses are stored here.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SnapshotOut {
    pub block_number: u64,
    pub n: usize,
    pub logn: usize,
    pub srs_id: String,
    pub gc_hex: String,

    pub universe_mode: String,
    pub balance_encoding: String, // e.g. "fr_exact_from_u256"
}

/// Publisher update record per block: (β, Δ) and pinned commitment C_t.
/// Paper-faithful publisher publishes commitments and updates.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct JournalLine {
    pub block_number: u64,
    pub n: usize,
    pub logn: usize,
    pub srs_id: String,

    /// β indices updated at this block (sorted, unique)
    pub changed_indices: Vec<usize>,
    /// Δ values (Fr) aligned with changed_indices
    pub delta_hex: Vec<String>,

    /// Publisher pins C_t for this block
    #[serde(default)]
    pub gc_hex: Option<String>,
}

/// Single point proof (witness) for one index in α
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SingleProof {
    pub index: usize,      // i in [0..N)
    pub address: String,   // 0x...
    pub value_hex: String, // Fr compressed hex
    pub gq_hex: String,    // G1 compressed hex
}

/// User-maintained state (paper-faithful):
/// user stores ONLY what they need for their α:
/// - α indices / addresses
/// - values v_α
/// - witnesses W_α
/// - current commitment C_t and last synced block
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UserState {
    pub n: usize,
    pub logn: usize,
    pub srs_id: String,

    pub last_block: u64,
    pub gc_hex: String,

    pub alpha_indices: Vec<usize>,
    pub alpha_addresses: Vec<String>,

    pub alpha_values_hex: Vec<String>,
    pub alpha_witnesses_hex: Vec<String>, // G1 compressed hex, aligned with alpha_indices

    pub universe_mode: String,
    pub value_mode: String, // "user_maintains_values_forward"
    pub witness_mode: String, // "user_maintains_witnesses_vupdate"
}

/// User query output at a given block
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QueryAtOut {
    pub query_block: u64,
    pub n: usize,
    pub logn: usize,
    pub srs_id: String,
    pub gc_hex: String,

    pub indices: Vec<usize>,
    pub addresses: Vec<String>,
    pub values_hex: Vec<String>,

    pub single_proofs: Vec<SingleProof>,
    pub aggregated_proof_hex: String,
    pub verify_multi_ok: bool,

    pub engine: String,
}

/// Evaluation output for historical range queries (optional)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HistoricalRangeQueryOut {
    pub block_range: (u64, u64),
    pub total_blocks: usize,
    pub step: u64,

    pub alpha_count: usize,
    pub queries: Vec<QueryAtOut>,
    pub stats: RangeQueryStats,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RangeQueryStats {
    pub total_commitments: usize,
    pub total_aggregated_proofs: usize,
    pub total_single_proofs: usize,
    pub total_values: usize,
    pub successful_verifications: usize,
    pub failed_verifications: usize,
    pub avg_verification_time_us: Option<u128>,
}
