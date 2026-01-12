// src/dataset_helpers.rs
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use ark_bls12_381::fr::Fr;
use ark_ff::Zero;
use ethers_core::types::{Address, U256};
use eyre::{bail, Result};

use crate::codec::fr_from_u256_exact;
use crate::dataset::DatasetReader;
use crate::types::SnapshotState;

/// Tracks exact Fr balances for a fixed universe.
///
/// - Initialized from snapshot_state.json at B0 (U256 balances converted to Fr exactly),
///   OR initialized from snapshot_vals_u256hex.txt aligned with universe.txt order.
/// - Then advanced only by dataset blocks B0+1...
pub struct StateTracker {
    pub cur_block: u64,
    balances: HashMap<Address, Fr>,
    universe: HashSet<Address>,
    dataset: DatasetReader,
}

impl StateTracker {
    /// Initialize from application snapshot_state.json (address->decimal string U256).
    ///
    /// NOTE: uses universe_addrs for deterministic coverage and nicer errors.
    pub fn from_snapshot_file(
        dataset_dir: &Path,
        segment_size: u32,
        snapshot_state_path: &Path,
        universe_addrs: &[Address],
    ) -> Result<Self> {
        let bytes = fs::read(snapshot_state_path)?;
        let st: SnapshotState = serde_json::from_slice(&bytes)?;

        if st.balance_encoding != "u256" {
            bail!(
                "SnapshotState.balance_encoding must be 'u256', got {}",
                st.balance_encoding
            );
        }

        let universe: HashSet<Address> = universe_addrs.iter().copied().collect();
        let mut balances: HashMap<Address, Fr> = HashMap::with_capacity(universe.len());

        for &a in universe_addrs {
            let key = format!("{:#x}", a);
            let bal_dec = st
                .balances
                .get(&key)
                .ok_or_else(|| eyre::eyre!("snapshot_state missing universe address {}", key))?;
            let u = U256::from_dec_str(bal_dec)?;
            let f = fr_from_u256_exact(u)?;
            balances.insert(a, f);
        }

        Ok(Self {
            cur_block: st.block_number,
            balances,
            universe,
            dataset: DatasetReader::new(dataset_dir, segment_size),
        })
    }

    /// Initialize from snapshot_vals file aligned with universe order.
    ///
    /// snapshot_vals format:
    /// - one value per line, either "0x..." hex OR decimal
    /// - line count must equal universe_addrs.len()
    pub fn from_snapshot_vals_file(
        dataset_dir: &Path,
        segment_size: u32,
        snapshot_block: u64,
        universe_addrs: &[Address],
        snapshot_vals_path: &Path,
    ) -> Result<Self> {
        let text = fs::read_to_string(snapshot_vals_path)?;

        let universe: HashSet<Address> = universe_addrs.iter().copied().collect();
        let mut balances: HashMap<Address, Fr> = HashMap::with_capacity(universe.len());

        let mut line_iter = text.lines();

        for (i, addr) in universe_addrs.iter().copied().enumerate() {
            let line = line_iter
                .next()
                .ok_or_else(|| eyre::eyre!("snapshot_vals ended early at line {}", i + 1))?;
            let s = line.trim();
            if s.is_empty() {
                bail!("snapshot_vals: empty line at {}", i + 1);
            }

            let u = if let Some(hex) = s.strip_prefix("0x") {
                U256::from_str_radix(hex, 16).map_err(|e| {
                    eyre::eyre!("snapshot_vals bad hex at line {}: {} ({})", i + 1, s, e)
                })?
            } else {
                U256::from_dec_str(s).map_err(|e| {
                    eyre::eyre!("snapshot_vals bad dec at line {}: {} ({})", i + 1, s, e)
                })?
            };

            let f = fr_from_u256_exact(u)?;
            balances.insert(addr, f);
        }

        // Ensure no extra non-empty lines
        for (j, extra) in line_iter.enumerate() {
            if extra.trim().is_empty() {
                continue;
            }
            bail!(
                "snapshot_vals has extra data after {} lines (first extra at line {}: {})",
                universe_addrs.len(),
                universe_addrs.len() + j + 1,
                extra.trim()
            );
        }

        Ok(Self {
            cur_block: snapshot_block,
            balances,
            universe,
            dataset: DatasetReader::new(dataset_dir, segment_size),
        })
    }

    pub fn get_balance(&self, addr: &Address) -> Fr {
        self.balances.get(addr).copied().unwrap_or(Fr::zero())
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

            let oldf = self.balances.get(&e.address).copied().unwrap_or(Fr::zero());
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
