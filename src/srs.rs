// src/srs.rs
use crate::codec::keccak_hex;
use crate::vc_context::VcContext;
use crate::vc_parameter::VcParameter;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::{Result, bail};
use std::{fs, path::Path};

pub fn load_or_create_srs(path: &Path, logn: usize) -> Result<(VcParameter, String)> {
    eprintln!("SRS: requested path={:?}, logn={}", path, logn);

    if path.exists() {
        eprintln!("SRS: found existing SRS at {:?}, loading…", path);
        let bytes = fs::read(path)?;
        let vp: VcParameter = VcParameter::deserialize_compressed(&*bytes)
            .map_err(|_| eyre::eyre!("SRS file corrupt"))?;
        if vp.logn != logn {
            bail!("SRS logn={} != {}", vp.logn, logn);
        }
        let mut ser = Vec::new();
        vp.serialize_compressed(&mut ser)?;
        let srs_id = keccak_hex(&ser);
        eprintln!("SRS: loaded existing SRS, id={}", srs_id);
        return Ok((vp, srs_id));
    }

    // create
    eprintln!("SRS: {:?} not found, generating new SRS (logn={})…", path, logn);
    let mut rng = ark_std::test_rng();

    let (_trap, vp) = VcParameter::new(logn, &mut rng);

    eprintln!("SRS: finished VcParameter::new, serializing…");
    let mut ser = Vec::new();
    vp.serialize_compressed(&mut ser)?;
    eprintln!(
        "SRS: serialization done ({} bytes), writing to {:?}…",
        ser.len(),
        path
    );

    fs::write(path, &ser)?;
    let srs_id = keccak_hex(&ser);
    eprintln!("SRS: done, wrote SRS to {:?}, id={}", path, srs_id);

    Ok((vp, srs_id))
}

pub fn make_ctx(vp: &VcParameter, logn: usize) -> VcContext {
    VcContext::new(vp, logn)
}
