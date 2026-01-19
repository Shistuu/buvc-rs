use ark_bls12_381::fr::Fr;
use ark_ff::Zero;

use std::{collections::{HashMap, HashSet}, fs, path::Path};

use ethers_core::types::{Address, U256};

use eyre::{bail, Result};

use crate::{codec::fr_from_u256_exact, types::{DatasetReader, StateTracker}};

impl StateTracker {
    /// Initialize from snapshot values file
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

    /// Apply changes from next block and return balance deltas
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
