// src/indexer.rs
use ethers_core::types::Address;
use eyre::{bail, Result};
use std::collections::HashMap;
use crate::types::Indexer;

impl Indexer {
    /// Collision-free: idx = position in sorted universe list.
    pub fn from_universe_sequential(universe_addrs: &[Address], n: usize) -> Result<Self> {
        if n == 0 {
            bail!("N must be > 0");
        }

        let mut sorted = universe_addrs.to_vec();
        sorted.sort();
        sorted.dedup();

        if sorted.len() > n {
            bail!("universe size {} > N {}. Increase logn.", sorted.len(), n);
        }

        let mut addr_to_idx = HashMap::with_capacity(sorted.len());
        for (i, a) in sorted.into_iter().enumerate() {
            addr_to_idx.insert(a, i);
        }
        Ok(Self { n, addr_to_idx })
    }

    #[inline]
    pub fn index_of(&self, a: Address) -> Result<usize> {
        self.addr_to_idx
            .get(&a)
            .copied()
            .ok_or_else(|| eyre::eyre!("address not in universe: {:#x}", a))
    }
}
