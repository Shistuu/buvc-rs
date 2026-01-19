use std::collections::HashMap;

use eyre::{bail, Result};

use crate::types::{UserState, ProofServerUser, ProofServerState};

impl ProofServerState {
    pub fn from_user_states(users: &[(String, UserState)]) -> Result<Self> {
        if users.is_empty() {
            bail!("ProofServerInit: no users");
        }

        let base = &users[0].1;

        // Collect union α and one witness per index (all users aligned at same block)
        let mut union_alpha: Vec<usize> = Vec::new();
        let mut wit_map: HashMap<usize, String> = HashMap::new();

        for (uid, u) in users {
            if u.n != base.n || u.logn != base.logn || u.srs_id != base.srs_id {
                bail!("user_state invariant mismatch for {}", uid);
            }
            if u.last_block != base.last_block || u.gc_hex != base.gc_hex {
                bail!("user_states not aligned at same block/commitment");
            }
            if u.alpha_indices.len() != u.alpha_witnesses_hex.len() {
                bail!("user_state α length mismatch for {}", uid);
            }

            for (k, &idx) in u.alpha_indices.iter().enumerate() {
                union_alpha.push(idx);
                wit_map.entry(idx).or_insert_with(|| u.alpha_witnesses_hex[k].clone());
            }
        }

        union_alpha.sort_unstable();
        union_alpha.dedup();

        let mut alpha_witnesses_hex = Vec::with_capacity(union_alpha.len());
        for &idx in &union_alpha {
            alpha_witnesses_hex.push(
                wit_map
                    .get(&idx)
                    .ok_or_else(|| eyre::eyre!("missing witness for union index {}", idx))?
                    .clone(),
            );
        }

        let users_out = users
            .iter()
            .map(|(id, u)| ProofServerUser {
                user_id: id.clone(),
                alpha_indices: u.alpha_indices.clone(),
            })
            .collect();

        Ok(Self {
            n: base.n,
            logn: base.logn,
            srs_id: base.srs_id.clone(),
            last_block: base.last_block,
            gc_hex: base.gc_hex.clone(),
            alpha_indices: union_alpha,
            alpha_witnesses_hex,
            users: users_out,
        })
    }

    /// Return (user_alpha_indices, positions in union-α vector)
    pub fn user_alpha(&self, user_id: &str) -> Option<(Vec<usize>, Vec<usize>)> {
        let user = self.users.iter().find(|u| u.user_id == user_id)?;

        // Fast pos map: index -> position
        let mut pos: HashMap<usize, usize> = HashMap::new();
        for (p, &idx) in self.alpha_indices.iter().enumerate() {
            pos.insert(idx, p);
        }

        let mut positions = Vec::with_capacity(user.alpha_indices.len());
        for &idx in &user.alpha_indices {
            let p = *pos.get(&idx)?;
            positions.push(p);
        }
        Some((user.alpha_indices.clone(), positions))
    }
}
