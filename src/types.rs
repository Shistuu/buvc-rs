use ark_bls12_381::{fr::Fr, G1Projective as G1};

use std::{collections::{HashMap, HashSet}, fs::File, path::PathBuf};

use clap::{Parser, Subcommand};

use ethers_core::types::{Address, U256};

use serde::{Deserialize, Serialize};

#[derive(serde::Serialize)]
pub struct Metric {
    pub phase: &'static str,
    pub block: u64,
    pub n: usize,
    pub alpha: usize,
    pub beta: usize,
    pub micros: u128,
}


#[derive(Parser, Debug)]
#[command(name = "cauchy-runner")]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    BuildUniverse {
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long)]
        start_block: u64,
        #[arg(long)]
        end_block: u64,
        #[arg(long)]
        out: PathBuf,
    },

    PublisherSnapshot {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        block: u64,
        #[arg(long)]
        universe_file: PathBuf,
        #[arg(long)]
        snapshot_vals: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },

    PublisherAdvance {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long)]
        universe_file: PathBuf,
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        snapshot_vals: PathBuf,
        #[arg(long)]
        end_block: u64,
        #[arg(long)]
        journal: PathBuf,
    },
    IssueUserState {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        universe_file: PathBuf,
        #[arg(long)]
        addresses: PathBuf,
        #[arg(long)]
        snapshot_vals: PathBuf, 
        #[arg(long)]
        out: PathBuf,
    },

       ProofServerInit {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        user_id: Vec<String>,
        #[arg(long)]
        user_state: Vec<PathBuf>,
    
        #[arg(long)]
        out: PathBuf,
    },
    
    ProofServerAdvance {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        proof_server_state: PathBuf,
        #[arg(long)]
        journal: PathBuf,
        #[arg(long)]
        end_block: u64,
        #[arg(long)]
        out: PathBuf,
    },

    ProofServerHistory {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        proof_server_state: PathBuf,
        #[arg(long)]
        journal: PathBuf,
        #[arg(long)]
        start_block: u64,
        #[arg(long)]
        blocks: Vec<u64>,
        #[arg(long)]
        user_id: String,
        #[arg(long)]
        out: PathBuf,
    },
    UserVerify {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        user_state: PathBuf,
        #[arg(long)]
        history: PathBuf,
        #[arg(long)]
        claims: Option<PathBuf>,
    },
    ProofServerHistoryServer {
        #[arg(long)]
        logn: usize,
        #[arg(long)]
        srs: PathBuf,
        #[arg(long)]
        proof_server_state: PathBuf,
        #[arg(long)]
        journal: PathBuf,
        #[arg(long)]
        r#in: Option<PathBuf>,
    },
    ExtractValue {
    #[arg(long)]
    dataset_dir: PathBuf,

    #[arg(long)]
    snapshot: PathBuf,

    #[arg(long)]
    snapshot_vals: PathBuf,

    #[arg(long)]
    universe_file: PathBuf,

    #[arg(long)]
    target_block: u64,

    #[arg(long)]
    index: usize,
},

    
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct JournalLineLite {
    pub block_number: u64,
    pub n: usize,
    pub logn: usize,
    pub srs_id: String,
    pub changed_indices: Vec<usize>,
    pub delta_hex: Vec<String>,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SnapshotState {
    pub block_number: u64,
    pub balances: std::collections::HashMap<String, String>,
    pub balance_encoding: String,
}

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
    pub changed_indices: Vec<usize>,
    pub delta_hex: Vec<String>,
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
    pub alpha_witnesses_hex: Vec<String>,
    pub universe_mode: String,
    pub witness_mode: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProofServerHistoryOut {
    pub user_id: String,
    pub block: u64,
    pub indices: Vec<usize>,
    pub witnesses_hex: Vec<String>,
    pub gc_hex: String,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ValueClaim {
    pub block: u64,
    pub values_hex: Vec<String>,
}

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

pub struct StateTracker {
    pub cur_block: u64,
    pub balances: HashMap<Address, Fr>,
    pub universe: HashSet<Address>,
    pub dataset: DatasetReader,
}

#[derive(Clone, Debug)]
pub struct Indexer {
    pub n: usize,
    pub addr_to_idx: HashMap<Address, usize>,
}

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
    pub last_block: u64,
    pub gc_hex: String,
    pub alpha_indices: Vec<usize>,
    pub alpha_witnesses_hex: Vec<String>,
    pub users: Vec<ProofServerUser>,
}

#[derive(Clone, Debug)]
pub enum HistoryOp {
    Update { beta: Vec<usize>, delta: Vec<Fr> },
    Query { query_id: usize },
}


#[derive(Clone, Debug)]
pub struct HistoryQueryResult {
    pub query_id: usize,
    pub indices: Vec<usize>,
    pub proofs: Vec<G1>,
}