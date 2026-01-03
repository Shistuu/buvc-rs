// src/dataset.rs
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ethers_core::types::{Address, U256};
use eyre::{bail, Result};

#[derive(Clone, Debug)]
pub struct Entry {
    pub address: Address,
    pub balance: U256,
}

pub struct DatasetReader {
    dir: PathBuf,
    segment_size: u32,
    current_segment: Option<SegmentReader>,
}

impl DatasetReader {
    pub fn new<P: AsRef<Path>>(dir: P, segment_size: u32) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            segment_size,
            current_segment: None,
        }
    }

    pub fn get_block(&mut self, block: u32) -> Result<Vec<Entry>> {
        self.ensure_segment(block)?;
        self.current_segment
            .as_mut()
            .ok_or_else(|| eyre::eyre!("Failed to open segment"))?
            .get_block(block)
    }

    pub fn iterate_range<F>(&mut self, start: u32, end: u32, mut f: F) -> Result<()>
    where
        F: FnMut(u32, Vec<Entry>) -> Result<()>,
    {
        if end < start {
            return Ok(());
        }

        let mut current = start;
        while current <= end {
            self.ensure_segment(current)?;

            let (seg_last, next_start) = {
                let segment = self.current_segment.as_ref().unwrap();
                let seg_end = segment.end.min(end);
                let seg_last = segment.last_written.min(seg_end);
                let next_start = if current <= end && seg_last < segment.end {
                    segment.end + 1
                } else {
                    seg_last + 1
                };
                (seg_last, next_start)
            };

            for block in current..=seg_last {
                let entries = self.get_block(block)?;
                f(block, entries)?;
            }

            current = next_start;
        }

        Ok(())
    }

    fn ensure_segment(&mut self, block: u32) -> Result<()> {
        let (base, end) = segment_bounds(block, self.segment_size);

        if let Some(ref seg) = self.current_segment {
            if seg.base == base {
                return Ok(());
            }
        }

        self.current_segment = Some(SegmentReader::open(&self.dir, base, end)?);
        Ok(())
    }
}

struct SegmentReader {
    base: u32,
    end: u32,
    dat_file: File,
    offsets: Vec<u64>,
    last_written: u32,
}

impl SegmentReader {
    fn open<P: AsRef<Path>>(dir: P, base: u32, end: u32) -> Result<Self> {
        let dat_path = dir.as_ref().join(segment_name(base, end, "dat"));
        let idx_path = dir.as_ref().join(segment_name(base, end, "idx"));

        let dat_file = File::open(&dat_path)
            .map_err(|e| eyre::eyre!("Failed to open {}: {}", dat_path.display(), e))?;

        let mut idx_file = File::open(&idx_path)
            .map_err(|e| eyre::eyre!("Failed to open {}: {}", idx_path.display(), e))?;

        let mut idx_bytes = Vec::new();
        idx_file.read_to_end(&mut idx_bytes)?;

        if idx_bytes.len() % 8 != 0 {
            bail!("Index file corrupt: size {} not multiple of 8", idx_bytes.len());
        }

        let n_offsets = idx_bytes.len() / 8;
        if n_offsets < 1 {
            bail!("Index file missing initial offset");
        }

        let mut offsets = Vec::with_capacity(n_offsets);
        for i in 0..n_offsets {
            let offset = u64::from_le_bytes(
                idx_bytes[i * 8..i * 8 + 8]
                    .try_into()
                    .map_err(|_| eyre::eyre!("Failed to read offset"))?,
            );
            offsets.push(offset);
        }

        let last_written = if n_offsets == 1 {
            base.saturating_sub(1)
        } else {
            base + (n_offsets - 2) as u32
        };

        Ok(Self {
            base,
            end,
            dat_file,
            offsets,
            last_written,
        })
    }

    fn get_block(&mut self, block: u32) -> Result<Vec<Entry>> {
        if block < self.base || block > self.end {
            bail!("Block {} out of segment range [{}, {}]", block, self.base, self.end);
        }
        if block > self.last_written {
            bail!("Block {} not built yet (last built {})", block, self.last_written);
        }

        let i = (block - self.base) as usize;
        let start_offset = self.offsets[i];
        let end_offset = self.offsets[i + 1];

        if end_offset < start_offset {
            bail!("Corrupt index: end < start");
        }

        if end_offset == start_offset {
            return Ok(Vec::new());
        }

        let size = (end_offset - start_offset) as usize;
        self.dat_file.seek(SeekFrom::Start(start_offset))?;

        let mut buf = vec![0u8; size];
        self.dat_file.read_exact(&mut buf)?;

        if buf.len() < 4 {
            bail!("Data too short for block count");
        }

        let k = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let mut pos = 4;

        let mut entries = Vec::with_capacity(k);
        for _ in 0..k {
            if pos + 20 > buf.len() {
                bail!("Data too short for address");
            }

            let mut addr_bytes = [0u8; 20];
            addr_bytes.copy_from_slice(&buf[pos..pos + 20]);
            let address = Address::from(addr_bytes);
            pos += 20;

            let (balance_len, n) = read_uvarint(&buf[pos..])?;
            pos += n;

            let balance = if balance_len > 0 {
                if pos + balance_len > buf.len() {
                    bail!("Data too short for balance");
                }
                if balance_len > 32 {
                    bail!("Balance too large: {} bytes", balance_len);
                }
                let mut balance_bytes = vec![0u8; 32];
                let start = 32 - balance_len;
                balance_bytes[start..].copy_from_slice(&buf[pos..pos + balance_len]);
                U256::from_big_endian(&balance_bytes)
            } else {
                U256::zero()
            };
            pos += balance_len;

            entries.push(Entry { address, balance });
        }

        Ok(entries)
    }
}

fn read_uvarint(buf: &[u8]) -> Result<(usize, usize)> {
    let mut result = 0u64;
    let mut shift = 0;
    let mut bytes_read = 0;

    for &byte in buf.iter().take(10) {
        bytes_read += 1;
        result |= ((byte & 0x7F) as u64) << shift;

        if (byte & 0x80) == 0 {
            return Ok((result as usize, bytes_read));
        }

        shift += 7;
        if shift >= 64 {
            bail!("uvarint too large");
        }
    }

    bail!("uvarint incomplete");
}

fn segment_bounds(block: u32, segment_size: u32) -> (u32, u32) {
    let base = (block / segment_size) * segment_size;
    let end = base + segment_size - 1;
    (base, end)
}

fn segment_name(base: u32, end: u32, ext: &str) -> String {
    format!("blk_{:08}_{:08}.{}", base, end, ext)
}
