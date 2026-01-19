// src/codec.rs
use ark_bls12_381::{fr::Fr, G1Projective as G1};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ethers_core::{
    types::{U256},
    utils::keccak256,
};
use eyre::{bail, Result};

pub fn fr_from_hex(hex_str: &str) -> Result<Fr> {
    let bytes = hex::decode(hex_str.trim_start_matches("0x"))?;
    let f = Fr::deserialize_compressed(&*bytes).map_err(|_| eyre::eyre!("bad Fr hex"))?;
    Ok(f)
}

pub fn fr_to_hex(f: &Fr) -> String {
    let mut bytes = Vec::new();
    f.serialize_compressed(&mut bytes).unwrap();
    format!("0x{}", hex::encode(bytes))
}

pub fn g1_from_hex(hex_str: &str) -> Result<G1> {
    let bytes = hex::decode(hex_str.trim_start_matches("0x"))?;
    let g = G1::deserialize_compressed(&*bytes)
        .map_err(|_| eyre::eyre!("bad G1 hex (cannot deserialize)"))?;
    Ok(g)
}

pub fn g1_to_hex(g: &G1) -> String {
    let mut bytes = Vec::new();
    g.serialize_compressed(&mut bytes).unwrap();
    format!("0x{}", hex::encode(bytes))
}

pub fn keccak_hex(bytes: &[u8]) -> String {
    format!("0x{}", hex::encode(keccak256(bytes)))
}

pub fn fr_from_u256_exact(x: U256) -> Result<Fr> {
    // U256 -> 32-byte big-endian
    let mut be = [0u8; 32];
    x.to_big_endian(&mut be);

    // Parse into 4 u64 big-endian words: w0=highest ... w3=lowest
    let w0 = u64::from_be_bytes(be[0..8].try_into().unwrap());
    let w1 = u64::from_be_bytes(be[8..16].try_into().unwrap());
    let w2 = u64::from_be_bytes(be[16..24].try_into().unwrap());
    let w3 = u64::from_be_bytes(be[24..32].try_into().unwrap());

    // arkworks BigInt uses little-endian u64 "words"
    type FrBigInt = <Fr as PrimeField>::BigInt;
    let words_le = [w3, w2, w1, w0];
    let bx = FrBigInt::new(words_le);

    // Reject values that would be reduced mod p
    if bx >= Fr::MODULUS {
        bail!("balance does not fit in Fr exactly (>= modulus); choose a different encoding");
    }

    // Safe because bx < MODULUS
    Ok(Fr::from_bigint(bx).expect("bx < MODULUS implies valid field element"))
}

/// delta = new - old in Fr
pub fn delta_fr(newv: Fr, oldv: Fr) -> Fr {
    newv - oldv
}
