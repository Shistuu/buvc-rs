// src/types.rs
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SnapshotState {
    pub block_number: u64,
    pub balances: std::collections::HashMap<String, String>,
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
    pub balance_encoding: String,
}

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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UserState {
    pub n: usize,
    pub logn: usize,
    pub srs_id: String,

    pub last_block: u64,
    pub gc_hex: String,

    pub alpha_indices: Vec<usize>,
    pub alpha_addresses: Vec<String>,

    pub alpha_witnesses_hex: Vec<String>, // G1 compressed hex, aligned with alpha_indices

    pub universe_mode: String,
    pub witness_mode: String, // "user_maintains_witnesses_vupdate"
}