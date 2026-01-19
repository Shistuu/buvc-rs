// src/types.rs
use serde::{Deserialize, Serialize};
use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ethers_core::types::{Address, U256};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::fs::File;

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

    pub alpha_witnesses_hex: Vec<String>, // G1 compressed hex, aligned with alpha_indices

    pub universe_mode: String,
    pub witness_mode: String, // "user_maintains_witnesses_vupdate"
}

// Dataset related
pub struct Entry {
    pub address: Address,
    pub balance: U256,
}

pub struct SegmentReader {
    pub base: u32,
    pub end: u32,
    pub dat_file: File,
    pub offsets: Vec<u64>,
    pub last_written: u32,
}

pub struct DatasetReader {
    pub dir: PathBuf,
    pub segment_size: u32,
    pub current_segment: Option<SegmentReader>,
}

// Dataset helpers
pub struct StateTracker {
    pub cur_block: u64,
    pub balances: HashMap<Address, Fr>,
    pub universe: HashSet<Address>,
    pub dataset: DatasetReader,
}

// Indexer
#[derive(Clone, Debug)]
pub struct Indexer {
    pub n: usize,
    pub addr_to_idx: HashMap<Address, usize>,
}

// Proof server
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
    /// Commitment at last_block (optional but useful for sanity checks)
    pub gc_hex: String,
    /// UNION α across all users
    pub alpha_indices: Vec<usize>,
    /// Witnesses for UNION α at last_block
    pub alpha_witnesses_hex: Vec<String>,
    /// Users
    pub users: Vec<ProofServerUser>,
}

// History
#[derive(Clone, Debug)]
pub enum HistoryOp {
    Update { beta: Vec<usize>, delta: Vec<Fr> },
    Query { query_id: usize },
}

#[derive(Clone, Debug)]
pub struct HistoryQueryResult {
    pub query_id: usize,
    pub indices: Vec<usize>, // α
    pub proofs: Vec<G1>,     // witnesses for α, same order
}