// src/snapshot_vals.rs
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use ethers_core::types::U256;
use eyre::{bail, Result};

pub fn read_u256hex_lines(path: &Path) -> Result<Vec<U256>> {
    let f = File::open(path)?;
    let rd = BufReader::new(f);
    let mut out = Vec::new();

    for (i, line) in rd.lines().enumerate() {
        let s = line?;
        let s = s.trim();
        if s.is_empty() {
            bail!("empty balance line {}", i + 1);
        }
        // Accept "0x..." hex
        let s = s.strip_prefix("0x").unwrap_or(s);
        let u = U256::from_str_radix(s, 16)
            .map_err(|e| eyre::eyre!("bad U256 hex at line {}: {} ({})", i + 1, s, e))?;
        out.push(u);
    }

    Ok(out)
}
