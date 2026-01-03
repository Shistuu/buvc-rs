// src/erigon.rs
use std::collections::HashMap;
use std::str::FromStr;

use ethers_core::types::{Address, BlockId, BlockNumber, U256};
use ethers_providers::{Ipc, Middleware, Provider};

use eyre::Result;
use serde::Deserialize;
use serde_json::json;


#[derive(Debug, Deserialize)]
struct AccountRangeAccount {
    address: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AccountRangeResult {
    accounts: HashMap<String, AccountRangeAccount>,
    next: Option<String>,
}

/// In synthetic mode returns Ok(None) so callers can skip IPC entirely.
/// In real mode returns Ok(Some(Provider<Ipc>)).
pub async fn provider_via_ipc(ipc_path: &str) -> Result<Option<Provider<Ipc>>> {
    let ipc_client = Ipc::connect(ipc_path.to_string()).await?;
    Ok(Some(Provider::new(ipc_client)))
}

/// Unified balance fetcher:
/// - If `CAUCHY_SYNTH_SEED` is set, returns deterministic balances without using the provider.
/// - Otherwise uses the given provider (must be Some) to query real balances.
pub async fn fetch_balances_maybe_synth(
    provider: &Provider<Ipc>,
    addrs: &[Address],
    block_num: u64,
) -> Result<Vec<U256>> {
    let bid: BlockId = BlockNumber::Number(block_num.into()).into();
    let mut v = Vec::with_capacity(addrs.len());
    for a in addrs {
        v.push(provider.get_balance(*a, Some(bid)).await?);
    }
    Ok(v)
}
/// Only valid in real mode. Callers must ensure they pass Some(provider) when they want state enumeration.
pub async fn dump_universe_at_block(
    provider: &Provider<Ipc>,
    block_num: u64,
    page: usize,
) -> eyre::Result<Vec<Address>> {
    let mut out = Vec::<Address>::new();
    let mut start: Option<String> = None;

    loop {
        // Params: (blockHex, startKey, maxResults, preimages, accounts)
        let params = json!([
            format!("0x{:x}", block_num),
            start.clone().unwrap_or_default(),
            page,
            true,  // preimages
            true,  // accounts
        ]);

        // Some rpcdaemon variants wrap the result, so request as Value first.
        let v: serde_json::Value = provider.request("debug_accountRange", params).await?;
        let res_val = v.get("result").cloned().unwrap_or(v);
        let res: AccountRangeResult = serde_json::from_value(res_val)?;

        for (_k, acct) in res.accounts {
            if let Some(addr_hex) = acct.address {
                if let Ok(a) = Address::from_str(&addr_hex) {
                    out.push(a);
                }
            }
        }

        if let Some(next) = res.next {
            if next.is_empty() {
                break;
            }
            start = Some(next);
        } else {
            break;
        }
    }

    out.sort_unstable();
    out.dedup();
    Ok(out)
}
