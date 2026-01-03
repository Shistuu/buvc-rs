// src/dataset_helpers.rs
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use ark_bls12_381::fr::Fr;
use ark_ff::Field;
use ethers_core::types::{Address, U256};
use eyre::{bail, Result};

use crate::codec::fr_from_u256_exact;
use crate::dataset::DatasetReader;
use crate::types::SnapshotState;

/// Tracks exact Fr balances for a fixed universe.
/// Initialized from snapshot_state.json at B0 (U256 balances converted to Fr exactly).
/// Then advanced only by dataset blocks B0+1...
pub struct StateTracker {
    pub cur_block: u64,
    balances: HashMap<Address, Fr>,
    universe: HashSet<Address>,
    dataset: DatasetReader,
}

impl StateTracker {
    pub fn from_snapshot_file(
        dataset_dir: &Path,
        segment_size: u32,
        snapshot_state_path: &Path,
        universe: HashSet<Address>,
    ) -> Result<Self> {
        let bytes = fs::read(snapshot_state_path)?;
        let st: SnapshotState = serde_json::from_slice(&bytes)?;

        if st.balance_encoding != "u256" {
            bail!(
                "SnapshotState.balance_encoding must be 'u256', got {}",
                st.balance_encoding
            );
        }

        let mut balances: HashMap<Address, Fr> = HashMap::with_capacity(universe.len());
        for a in &universe {
            let key = format!("{:#x}", a);
            let bal_dec = st
                .balances
                .get(&key)
                .ok_or_else(|| eyre::eyre!("snapshot_state missing universe address {}", key))?;
            let u = U256::from_dec_str(bal_dec)?;
            let f = fr_from_u256_exact(u)?;
            balances.insert(*a, f);
        }

        Ok(Self {
            cur_block: st.block_number,
            balances,
            universe,
            dataset: DatasetReader::new(dataset_dir, segment_size),
        })
    }

    pub fn get_balance(&self, addr: &Address) -> Fr {
        self.balances.get(addr).copied().unwrap_or(Fr::ZERO)
    }

    /// Applies exactly one block (must be cur_block+1) and updates tracker balances.
    /// Returns (addr, old_fr, new_fr) only for universe addresses that changed.
    pub fn apply_next_block(&mut self) -> Result<Vec<(Address, Fr, Fr)>> {
        let b = self.cur_block + 1;
        let entries = self.dataset.get_block(b as u32)?;

        let mut changes = Vec::new();

        for e in entries {
            if !self.universe.contains(&e.address) {
                continue;
            }

            let oldf = self.balances.get(&e.address).copied().unwrap_or(Fr::ZERO);
            let newf = fr_from_u256_exact(e.balance)?;

            if oldf != newf {
                changes.push((e.address, oldf, newf));
                self.balances.insert(e.address, newf);
            }
        }

        self.cur_block = b;
        Ok(changes)
    }
}
