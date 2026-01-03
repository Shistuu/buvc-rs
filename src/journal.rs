// src/journal.rs
use crate::types::JournalLine;
use eyre::Result;
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{BufRead, Write},
    path::Path,
};

pub fn append_line<P: AsRef<Path>>(path: P, line: &JournalLine) -> Result<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", serde_json::to_string(line)?)?;
    Ok(())
}

pub fn iter_lines_filtered<P: AsRef<Path>>(
    path: P,
    srs_id: &str,
    n: usize,
    logn: usize,
) -> Result<Vec<JournalLine>> {
    if !path.as_ref().exists() {
        return Ok(Vec::new());
    }
    let f = std::fs::File::open(path)?;
    let rd = std::io::BufReader::new(f);
    let mut out = Vec::new();
    for line in rd.lines() {
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        let j: JournalLine = serde_json::from_str(&l)?;
        if j.srs_id == srs_id && j.n == n && j.logn == logn {
            out.push(j);
        }
    }
    Ok(out)
}

/// Return whether a journal entry for `block` already exists.
/// If both existing and new are pinned (Some), enforce equality.
pub fn already_pinned_block(
    lines: &[JournalLine],
    block: u64,
    gc_hex: &Option<String>,
) -> eyre::Result<bool> {
    for j in lines {
        if j.block_number == block {
            match (&j.gc_hex, gc_hex) {
                (Some(existing), Some(new)) => {
                    if existing != new {
                        eyre::bail!(
                            "journal already pinned block {} with different gc_hex.\nexisting:{}\nnew:{}",
                            block,
                            existing,
                            new
                        );
                    }
                }
                _ => { /* unpinned => don't enforce */ }
            }
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn pinned_gc_at(lines: &[JournalLine], block: u64) -> Option<String> {
    for j in lines {
        if j.block_number == block {
            return j.gc_hex.clone();
        }
    }
    None
}

/// Build a block->JournalLine map for fast lookup (forward or reverse).
pub fn map_by_block(lines: Vec<JournalLine>) -> BTreeMap<u64, JournalLine> {
    let mut m = BTreeMap::new();
    for j in lines {
        m.insert(j.block_number, j);
    }
    m
}
