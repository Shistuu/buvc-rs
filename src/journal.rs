use std::{fs::{File, OpenOptions}, io::{BufRead, BufReader, Write}, path::Path};

use eyre::Result;

use crate::types::JournalLine;

pub fn append_line<P: AsRef<Path>>(path: P, line: &JournalLine) -> Result<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", serde_json::to_string(line)?)?;
    Ok(())
}


/// Stream filtered journal lines without loading into memory
pub fn for_each_line_filtered<F>(
    journal_path: &Path,
    srs_id: &str,
    n: usize,
    logn: usize,
    mut f: F,
) -> Result<()>
where
    F: FnMut(JournalLine) -> Result<()>,
{
    if !journal_path.exists() {
        return Ok(());
    }
    let file = File::open(journal_path)?;
    let rd = BufReader::new(file);

    for line in rd.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let j: JournalLine = serde_json::from_str(line)?;
        if j.srs_id != srs_id || j.n != n || j.logn != logn {
            continue;
        }

        f(j)?;
    }
    Ok(())
}
