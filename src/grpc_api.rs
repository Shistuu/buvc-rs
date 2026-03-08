use eyre::Result;

use crate::types::ProofServerState;

/// Head point for the currently loaded tracked set:
/// returns (head_block, gc_hex, alpha_indices, alpha_witnesses_hex)
pub fn head_point_loaded(
    ps: &ProofServerState,
) -> Result<(u64, String, Vec<usize>, Vec<String>)> {
    if ps.alpha_indices.len() != ps.alpha_witnesses_hex.len() {
        return Err(eyre::eyre!(
            "alpha_indices / alpha_witnesses_hex length mismatch"
        ));
    }

    Ok((
        ps.last_block,
        ps.gc_hex.clone(),
        ps.alpha_indices.clone(),
        ps.alpha_witnesses_hex.clone(),
    ))
}