use std::iter;

use ark_bls12_381::fr::Fr;
use ark_bls12_381::Bls12_381;
use ark_bls12_381::G1Projective as G1;
use ark_bls12_381::G2Projective as G2;
use ark_ec::pairing::Pairing;
use ark_ec::Group;
use ark_ff::fields::FftField;
use ark_ff::fields::Field;
use ark_ff::Zero;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use ark_std::iterable::Iterable;

use crate::poly::compute_unity;
use crate::poly::poly_delta;
use crate::poly::poly_evaluate;
use crate::poly::poly_fft;
use crate::poly::poly_ifft;
use crate::poly::poly_interpolate;
use crate::poly::poly_zero;
use crate::vc_parameter::VcParameter;

/// Precomputed context in VC
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct VcContext {
    pub n: usize,
    pub logn: usize,
    /// n as a field element
    pub nf: Fr,
    /// Root of unity w^0, w^1, ..., w^(n-1)
    pub unity: Vec<Fr>,
    /// Group element L_0, L_1, ..., L_(n-1), as described in the paper
    pub gl: Vec<G1>,
    /// Group element L'_0, L'_1, ..., L'_(n-1), as described in the paper
    pub gll: Vec<G1>,
}

impl VcContext {
    pub fn new(vc_p: &VcParameter, logn: usize) -> VcContext {
        let n = 1 << logn;
        let gs = &vc_p.gs1;

        assert!(n <= vc_p.n);
        assert!(logn <= Fr::TWO_ADICITY as usize);
        let unity: Vec<Fr> = compute_unity(n << 1);

        let mut gl: Vec<G1> = (0..n).map(|i| gs[n - i - 1]).collect();
        poly_fft(&unity, &mut gl, n);

        let mut gll: Vec<G1> = (0..n - 1)
            .map(|i| gs[n - i - 2] * Fr::from((i + 1) as u32))
            .chain(iter::once(G1::zero()))
            .collect();
        poly_fft(&unity, &mut gll, n);

        VcContext {
            n,
            logn,
            nf: Fr::from(n as u32),
            unity,
            gl,
            gll,
        }
    }

    /// Single-point verification
    pub fn verify(&self, vc_p: &VcParameter, gc: G1, index: usize, value: Fr, gq: G1) -> bool {
        let n = self.n;
        let unity = &self.unity;
        let gs2 = &vc_p.gs2;
        let step = unity.len() / n;

        let lhs = Bls12_381::pairing(gq, gs2[1] - G2::generator() * unity[index * step]);
        let rhs = Bls12_381::pairing(gc - G1::generator() * value, G2::generator());

        lhs == rhs
    }

    /// Multi-point verification
    pub fn verify_multi(
        &self,
        vc_p: &VcParameter,
        gc: G1,
        index: &[usize],
        value: &[Fr],
        gq: G1,
    ) -> bool {
        let n = self.n;
        let unity = &self.unity;
        let gs1 = &vc_p.gs1;
        let gs2 = &vc_p.gs2;
        let step = unity.len() / n;

        let x = &index.iter().map(|i| unity[*i * step]).collect::<Vec<Fr>>();
        let z = poly_zero(unity, x);
        let lag = poly_interpolate(unity, x, value);

        let lhs = Bls12_381::pairing(gq, (0..z.len()).map(|i| gs2[i] * z[i]).sum::<G2>());
        let rhs = Bls12_381::pairing(
            gc - (0..lag.len()).map(|i| gs1[i] * lag[i]).sum::<G1>(),
            G2::generator(),
        );

        lhs == rhs
    }

    /// Build a commitment and witnesses from v
    pub fn build_commitment(&self, v: &[Fr]) -> (G1, Vec<G1>) {
        let n = v.len();
        assert!(self.n == n);

        let nf = self.nf;
        let unity = &self.unity;
        let gl = &self.gl;
        let gll = &self.gll;
        let step = unity.len() / n;

        let mut a: Vec<Fr> = (0..n).map(|i| v[i] * unity[i * step] / nf).collect();
        let mut b: Vec<G1> = (0..n)
            .map(|i| gl[i] * (v[i] * unity[i * step] / nf))
            .collect();
        let gc = b.iter().sum();
        poly_fft(unity, &mut a, n);
        poly_fft(unity, &mut b, n);
        let tau: Vec<Fr> = (0..n)
            .map(|i| (nf - Fr::from((i * 2 + 1) as u32)) / Fr::from(2))
            .collect();
        a = a.into_iter().zip(&tau).map(|(i, j)| i * j).collect();
        b = b.into_iter().zip(&tau).map(|(i, j)| i * j).collect();
        poly_ifft(unity, &mut a, n);
        poly_ifft(unity, &mut b, n);
        a = a
            .into_iter()
            .zip(unity.iter().step_by(step))
            .map(|(i, j)| i * (Fr::ONE / j))
            .collect();
        b = b
            .into_iter()
            .zip(unity.iter().step_by(step))
            .map(|(i, j)| i * (Fr::ONE / j))
            .collect();

        let gq: Vec<G1> = (0..n)
            .map(|i| gll[i] * (v[i] * unity[i * step] / nf) + gl[i] * a[i] - b[i])
            .collect();

        (gc, gq)
    }

    /// Update commitment with a new delta
    pub fn update_commitment(&self, gc: G1, index: usize, value: Fr) -> G1 {
        let n = self.n;
        let nf = self.nf;
        let unity = &self.unity;
        let step = unity.len() / n;
        let gl = &self.gl;

        gc + gl[index] * (value * unity[index * step] / nf)
    }

    pub fn update_witnesses_batch(
        &self,
        alpha: &[usize], // Indices of the proofs
        gq: &[G1],       // The G1 elements corresponding to the proofs
        beta: &[usize],  // Indices of the modifications
        value: &[Fr],    // The Fr elements corresponding to the modifications
    ) -> Vec<G1> {
        // Always sorting alpha and beta at the beginning
        let mut sorted_alpha: Vec<_> = alpha.iter().zip(0..alpha.len()).collect();
        let mut sorted_beta: Vec<_> = beta.iter().zip(0..beta.len()).collect();
        sorted_alpha.sort_by(|a, b| a.0.cmp(b.0));
        sorted_beta.sort_by(|a, b| a.0.cmp(b.0));

        let mut common_alpha = Vec::new(); // For matching alpha indices
        let mut common_beta = Vec::new(); // For matching beta indices
        let mut alpha_extra = Vec::new(); // For extra alpha indices
        let mut beta_extra = Vec::new(); // For extra beta indices

        let mut i = 0;
        let mut j = 0;

        // Optimized loop for partitioning alpha and beta
        while i < sorted_alpha.len() && j < sorted_beta.len() {
            if sorted_alpha[i].0 == sorted_beta[j].0 {
                common_alpha.push(sorted_alpha[i]);
                common_beta.push(sorted_beta[j]);
                i += 1;
                j += 1;
            } else if sorted_alpha[i].0 < sorted_beta[j].0 {
                alpha_extra.push(sorted_alpha[i]);
                i += 1;
            } else {
                beta_extra.push(sorted_beta[j]);
                j += 1;
            }
        }

        // Add remaining elements from `alpha` and `beta`
        if i < sorted_alpha.len() {
            alpha_extra.extend_from_slice(&sorted_alpha[i..]);
        }
        if j < sorted_beta.len() {
            beta_extra.extend_from_slice(&sorted_beta[j..]);
        }

        let mut result = gq.to_vec();

        if !common_alpha.is_empty() {
            let m_alpha = common_alpha
                .iter()
                .map(|(x, _)| **x)
                .collect::<Vec<usize>>();
            let m_gq = common_alpha
                .iter()
                .map(|(_, y)| result[*y])
                .collect::<Vec<G1>>();
            let m_value = common_beta
                .iter()
                .map(|(_, x)| value[*x])
                .collect::<Vec<Fr>>();
            let updated_gq = self.update_witnesses_batch_same(&m_alpha, &m_gq, &m_value);
            for i in common_alpha.iter().zip(updated_gq.iter()) {
                result[i.0.1] = *i.1;
            }
            if !beta_extra.is_empty() {
                let m_alpha = common_alpha
                    .iter()
                    .map(|(x, _)| **x)
                    .collect::<Vec<usize>>();
                let m_gq = common_alpha
                    .iter()
                    .map(|(_, y)| result[*y])
                    .collect::<Vec<G1>>();
                let m_beta = beta_extra.iter().map(|(x, _)| **x).collect::<Vec<usize>>();
                let m_value = beta_extra
                    .iter()
                    .map(|(_, x)| value[*x])
                    .collect::<Vec<Fr>>();
                let updated_gq =
                    self.update_witnesses_batch_different(&m_alpha, &m_gq, &m_beta, &m_value);
                for i in common_alpha.iter().zip(updated_gq.iter()) {
                    result[i.0.1] = *i.1;
                }
            }
        }

        if !alpha_extra.is_empty() {
            let m_alpha = alpha_extra.iter().map(|(x, _)| **x).collect::<Vec<usize>>();
            let m_gq = alpha_extra
                .iter()
                .map(|(_, y)| result[*y])
                .collect::<Vec<G1>>();
            let m_beta = beta;
            let m_value = value;
            let updated_gq =
                self.update_witnesses_batch_different(&m_alpha, &m_gq, m_beta, m_value);
            for i in alpha_extra.iter().zip(updated_gq.iter()) {
                result[i.0.1] = *i.1;
            }
        }

        // Return the final result vector after all updates
        result
    }

    pub fn update_witnesses_batch_same(&self, alpha: &[usize], gq: &[G1], value: &[Fr]) -> Vec<G1> {
        let nf = &self.nf;
        let unity = &self.unity;
        let step = unity.len() / self.n;

        //Compute coefficients of zeroing polynomial
        let walpha = alpha.iter().map(|i| unity[*i * step]).collect::<Vec<Fr>>();
        let g = poly_zero(unity, &walpha);

        //The derivative of the zeroing polynomial
        let g_prime = g
            .iter()
            .skip(1)
            .enumerate()
            .map(|(i, x)| x * &Fr::from((i + 1) as u32))
            .collect::<Vec<Fr>>();

        //Derivative of the derivative of the zeroing polynomial
        let g_double_prime = g_prime
            .iter()
            .skip(1)
            .enumerate()
            .map(|(i, x)| x * &Fr::from((i + 1) as u32))
            .collect::<Vec<Fr>>();

        // Evaluate g' and g'' at the alpha points
        let g_prime_walpha = poly_evaluate(unity, &g_prime, &walpha);
        let g_double_prime_walpha = poly_evaluate(unity, &g_double_prime, &walpha);

        //Compute ya and yb with the first derivative of the zeroing polynomial
        let y_g_prime_a: Vec<Fr> = value
            .iter()
            .zip(walpha.iter().zip(g_prime_walpha.iter()))
            .map(|(v, (u, a))| v * u / nf * a)
            .collect();

        let y_g_prime_b: Vec<G1> = y_g_prime_a
            .iter()
            .enumerate()
            .map(|(i, v)| self.gl[alpha[i]] * v)
            .collect();

        //Compute ya and yb with the second derivative of the zeroing polynomial
        let y_g_double_prime_a: Vec<Fr> = value
            .iter()
            .zip(walpha.iter().zip(g_double_prime_walpha.iter()))
            .map(|(v, (u, a))| v * u / nf * a)
            .collect();

        let y_g_double_prime_b: Vec<G1> = y_g_double_prime_a
            .iter()
            .enumerate()
            .map(|(i, v)| self.gl[alpha[i]] * v)
            .collect();

        //Interpolate the polynomial to get ha and hb
        let h_a = poly_interpolate(&self.unity, &walpha, &y_g_prime_a);
        let h_b = poly_interpolate(&self.unity, &walpha, &y_g_prime_b);

        //Derivative of polinomial ha and hb
        let h_a_prime: Vec<Fr> = h_a
            .iter()
            .skip(1)
            .enumerate()
            .map(|(i, x)| x * &Fr::from((i + 1) as u32))
            .collect::<Vec<Fr>>();

        let h_b_prime: Vec<G1> = h_b
            .iter()
            .skip(1)
            .enumerate()
            .map(|(i, x)| *x * Fr::from((i + 1) as u32))
            .collect();

        //Compute coefficients of the derivate of ha and hb
        let eval_ha_prime = poly_evaluate(&self.unity, &h_a_prime, &walpha);
        let eval_hb_prime = poly_evaluate(&self.unity, &h_b_prime, &walpha);

        //Compute the first terms of the update
        let a_numerator: Vec<Fr> = eval_ha_prime
            .iter()
            .zip(y_g_double_prime_a.iter())
            .map(|(ha, y)| *ha - (*y / Fr::from(2u32)))
            .collect();

        let b_numerator: Vec<G1> = eval_hb_prime
            .iter()
            .zip(y_g_double_prime_b.iter())
            .map(|(hb, yb)| hb - (*yb * (Fr::ONE / Fr::from(2u32))))
            .collect();

        // Collect lindex into a Vec<G1>
        let lindex = alpha.iter().map(|i| self.gl[*i]);
        let nq: Vec<G1> = gq
            .iter()
            .zip(a_numerator.iter().zip(b_numerator.iter()))
            .zip(g_prime_walpha.iter().zip(lindex))
            .map(|((g, (a, b)), (z, l))| g + l * (a / z) - *b * (Fr::ONE / z))
            .collect();

        let llindex = alpha.iter().map(|i| self.gll[*i] * (unity[*i * step] / nf));
        nq.iter()
            .zip(llindex.zip(value.iter()))
            .map(|(n, (l, v))| n + l * v)
            .collect()
    }

    /// Update witnesses `gq` with index array `alpha`
    /// given updates on index `beta` and delta value `value`
    /// elements in `beta` must be different from `alpha`
    pub fn update_witnesses_batch_different(
        &self,
        alpha: &[usize],
        gq: &[G1],
        beta: &[usize],
        value: &[Fr],
    ) -> Vec<G1> {
        let nf = &self.nf;
        let unity = &self.unity;
        let step = unity.len() / self.n;

        let walpha = alpha.iter().map(|i| unity[*i * step]).collect::<Vec<Fr>>();
        let wbeta = beta.iter().map(|i| unity[*i * step]).collect::<Vec<Fr>>();

        let z = poly_zero(unity, &wbeta);

        let zz = z
            .iter()
            .skip(1)
            .enumerate()
            .map(|(i, x)| x * &Fr::from((i + 1) as u32))
            .collect::<Vec<Fr>>();

        let zwalpha = poly_evaluate(unity, &z, &walpha);
        let zzwbeta = poly_evaluate(unity, &zz, &wbeta);

        let ya: Vec<Fr> = value
            .iter()
            .zip(wbeta.iter().zip(zzwbeta.iter()))
            .map(|(v, (u, a))| v * u / nf * a)
            .collect();
        let yb: Vec<G1> = ya
            .iter()
            .enumerate()
            .map(|(i, v)| self.gl[beta[i]] * v)
            .collect();

        let ha = poly_interpolate(&self.unity, &wbeta, &ya);
        let hb = poly_interpolate(&self.unity, &wbeta, &yb);

        let evala = poly_evaluate(&self.unity, &ha, &walpha);
        let evalb = poly_evaluate(&self.unity, &hb, &walpha);

        let lindex = alpha.iter().map(|i| self.gl[*i]);
        gq.iter()
            .zip(evala.iter().zip(evalb.iter()))
            .zip(zwalpha.iter().zip(lindex))
            .map(|((g, (a, b)), (z, l))| g + l * (a / z) - *b * (Fr::ONE / z))
            .collect()
    }

    pub fn update_witness(&self, alpha: usize, gq: G1, beta: usize, value: Fr) -> G1 {
        let nf = &self.nf;
        let unity = &self.unity;
        let step = unity.len() / self.n;

        let walpha = unity[alpha * step];
        let wbeta = unity[beta * step];

        if alpha != beta {
            gq + (self.gl[alpha] - self.gl[beta]) * (value * wbeta / nf / (walpha - wbeta))
        } else {
            gq + self.gll[alpha] * (value * walpha / nf)
        }
    }

    /// Produce a multi-proof given a list of proofs
    pub fn aggregate_proof(&self, index: &[usize], gq: &[G1]) -> G1 {
        let n = self.n;
        let unity = &self.unity;
        let step = unity.len() / n;
        let x = &index.iter().map(|i| unity[*i * step]).collect::<Vec<Fr>>();
        let d = poly_delta(unity, x);
        (0..index.len()).map(|i| gq[i] * (Fr::ONE / d[i])).sum()
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::*;
    use crate::vc_parameter::tests::test_parameter;

    #[test]
    fn test_vc_context_new() {
        let (_s, vc_p) = test_parameter(3);
        let vc_c = VcContext::new(&vc_p, vc_p.logn);

        let n = vc_c.n;
        let gs1 = &vc_p.gs1;
        let unity = &vc_c.unity;
        let step = unity.len() / n;

        let mut gl: Vec<G1> = Vec::with_capacity(n);
        for i in 0..n {
            let mut x_power = Fr::ONE;
            gl.push(G1::zero());
            for j in 0..n {
                gl[i] += gs1[n - j - 1] * x_power;
                x_power *= unity[i * step];
            }
        }

        assert_eq!(gl, vc_c.gl);

        let mut gll: Vec<G1> = Vec::with_capacity(n);
        for i in 0..n {
            let mut x_power = Fr::ONE;
            gll.push(G1::zero());
            for j in 0..n - 1 {
                gll[i] += gs1[n - j - 2] * (Fr::from((j + 1) as u32) * x_power);
                x_power *= unity[i * step];
            }
        }

        assert_eq!(gll, vc_c.gll);
    }

    #[test]
    fn test_vc_context_commitment() {
        let (s, vc_p) = test_parameter(3);
        let vc_c = VcContext::new(&vc_p, 1);
        let n = vc_c.n;
        let v = [0, 1].map(Fr::from);
        let unity = &vc_c.unity;
        let step = unity.len() / n;

        let (gc, gq) = vc_c.build_commitment(&v);
        assert_eq!(
            gc,
            G1::generator()
                * (s * (Fr::ONE / (unity[step] - unity[0])) - (Fr::ONE / (unity[step] - unity[0])))
        );
        assert_eq!(
            gq[0],
            G1::generator() * (Fr::ONE / (unity[step] - unity[0]))
        );
        assert!(vc_c.verify(&vc_p, gc, 0, v[0], gq[0]));
        assert!(vc_c.verify(&vc_p, gc, 1, v[1], gq[1]));
    }

    #[test]
    fn test_vc_context_verify() {
        let (_s, vc_p) = test_parameter(3);
        let vc_c = VcContext::new(&vc_p, vc_p.logn);
        let v = [1, 4, 5, 2, 3, 6, 7, 0].map(Fr::from);

        let n = vc_c.n;
        let (gc, gq) = vc_c.build_commitment(&v);
        for i in 0..n {
            assert!(vc_c.verify(&vc_p, gc, i, v[i], gq[i]));
        }
        for i in 0..n {
            assert!(!vc_c.verify(&vc_p, gc, i, v[i] + Fr::ONE, gq[i]));
        }
    }

    #[test]
    fn test_vc_context_verify_multi() {
        let (_s, vc_p) = test_parameter(3);
        let vc_c = VcContext::new(&vc_p, vc_p.logn);
        let v = [1, 4, 5, 2, 3, 6, 7, 0].map(Fr::from);

        let (gc, gq) = vc_c.build_commitment(&v);
        let gqq = vc_c.aggregate_proof(&[0, 1, 2], &gq[0..=2]);
        assert!(vc_c.verify_multi(&vc_p, gc, &[0, 1, 2], &v[0..=2], gqq));

        let (gc, gq) = vc_c.build_commitment(&v);
        let gqq = vc_c.aggregate_proof(&[1, 2, 3], &gq[1..=3]);
        assert!(vc_c.verify_multi(&vc_p, gc, &[1, 2, 3], &v[1..=3], gqq));

        let (gc, gq) = vc_c.build_commitment(&v);
        let gqq = vc_c.aggregate_proof(&[1, 2, 3, 4], &gq[1..=4]);
        assert!(vc_c.verify_multi(&vc_p, gc, &[1, 2, 3, 4], &v[1..=4], gqq));

        let index = [3, 4, 5, 6];
        let gqq = vc_c.aggregate_proof(&index, &gq[3..=6]);
        assert!(vc_c.verify_multi(&vc_p, gc, &index, &v[3..=6], gqq));

        let index = [0, 1, 2, 3, 4, 5, 6];
        let gqq = vc_c.aggregate_proof(&index, &gq[0..=6]);
        assert!(vc_c.verify_multi(&vc_p, gc, &index, &v[0..=6], gqq));
    }

    #[test]
    fn test_update_witness() {
        let (_s, vc_p) = test_parameter(3);
        let vc_c = VcContext::new(&vc_p, vc_p.logn);

        let v = [1, 4, 5, 2, 3, 6, 7, 8].map(Fr::from); // Original values
        let vd = [11, 4, 25, 22, 3, 36, 7, 8].map(Fr::from); // Updated values

        let (gc, gq) = vc_c.build_commitment(&v);
        let (gcd, _gqd) = vc_c.build_commitment(&vd);

        let alpha = [1, 3, 5, 7];
        let beta = [0, 2, 3, 5];

        let mut updated_gq = [gq[1], gq[3], gq[5], gq[7]];
        let delta_value = [Fr::from(10), Fr::from(20), Fr::from(20), Fr::from(30)];

        for i in 0..alpha.len() {
            for j in 0..beta.len() {
                updated_gq[i] =
                    vc_c.update_witness(alpha[i], updated_gq[i], beta[j], delta_value[j]);
            }
        }

        let gc = vc_c.update_commitment(gc, beta[0], delta_value[0]);
        let gc = vc_c.update_commitment(gc, beta[1], delta_value[1]);
        let gc = vc_c.update_commitment(gc, beta[2], delta_value[2]);
        let gc = vc_c.update_commitment(gc, beta[3], delta_value[3]);

        assert!(gc == gcd);

        for i in 0..alpha.len() {
            assert!(vc_c.verify(&vc_p, gc, alpha[i], vd[alpha[i]], updated_gq[i]));
        }
    }

    #[test]
    fn test_update_witnesses_batch() {
        let (_s, vc_p) = test_parameter(3);
        let vc_c = VcContext::new(&vc_p, vc_p.logn);

        let v = [1, 4, 5, 2, 3, 6, 7, 8].map(Fr::from); // Original values
        let vd = [11, 4, 25, 22, 3, 36, 7, 8].map(Fr::from); // Updated values

        let (gc, gq) = vc_c.build_commitment(&v);
        let (gcd, _gqd) = vc_c.build_commitment(&vd);

        let alpha = [1, 3, 5, 7];
        let beta = [0, 2, 3, 5];

        let gq = [gq[1], gq[3], gq[5], gq[7]];
        let delta_value = [Fr::from(10), Fr::from(20), Fr::from(20), Fr::from(30)];

        let updated_gq = vc_c.update_witnesses_batch(&alpha, &gq, &beta, &delta_value);

        let gc = vc_c.update_commitment(gc, beta[0], delta_value[0]);
        let gc = vc_c.update_commitment(gc, beta[1], delta_value[1]);
        let gc = vc_c.update_commitment(gc, beta[2], delta_value[2]);
        let gc = vc_c.update_commitment(gc, beta[3], delta_value[3]);

        assert!(gc == gcd);

        for i in 0..alpha.len() {
            assert!(vc_c.verify(&vc_p, gc, alpha[i], vd[alpha[i]], updated_gq[i]));
        }
    }

    #[test]
    fn test_update_witnesses_batch_same() {
        let (_s, vc_p) = test_parameter(3);
        let vc_c = VcContext::new(&vc_p, vc_p.logn);
        let v = [1, 4, 5, 2, 3, 6, 7, 8].map(Fr::from);
        let vd = [1, 14, 5, 22, 3, 36, 7, 48].map(Fr::from);

        let (gc, gq) = vc_c.build_commitment(&v);
        let (gcd, _gqd) = vc_c.build_commitment(&vd);

        // Set up the indices for alpha and beta
        let alpha = [1, 3, 5, 7];
        let beta = [1, 3, 5, 7];
        let gq = [gq[1], gq[3], gq[5], gq[7]];
        let delta_value = [Fr::from(10), Fr::from(20), Fr::from(30), Fr::from(40)];

        // Update witnesses using batch same update
        let updated_gq = vc_c.update_witnesses_batch(&alpha, &gq, &alpha, &delta_value);
        let gc = vc_c.update_commitment(gc, beta[0], delta_value[0]);
        let gc = vc_c.update_commitment(gc, beta[1], delta_value[1]);
        let gc = vc_c.update_commitment(gc, beta[2], delta_value[2]);
        let gc = vc_c.update_commitment(gc, beta[3], delta_value[3]);

        assert!(gc == gcd);
        // Verify that the updated witnesses are correct
        for i in 0..alpha.len() {
            assert!(vc_c.verify(&vc_p, gc, alpha[i], vd[alpha[i]], updated_gq[i]));
        }
    }

    #[test]
    fn test_update_witnesses_batch_different() {
        let (_s, vc_p) = test_parameter(3);
        let vc_c = VcContext::new(&vc_p, vc_p.logn);
        let v = [1, 4, 5, 2, 3, 6, 7, 0].map(Fr::from);
        let vd = [11, 4, 25, 2, 3, 6, 7, 0].map(Fr::from);

        let (gc, gq) = vc_c.build_commitment(&v);
        let (gcd, _gqd) = vc_c.build_commitment(&vd);

        // Set up the indices for alpha and beta
        let alpha = [1, 3, 5];
        let beta = [0, 2];
        let gq = [gq[1], gq[3], gq[5]];
        let delta_value = [Fr::from(10), Fr::from(20)];

        // Update witnesses using batch different update
        let updated_gq = vc_c.update_witnesses_batch(&alpha, &gq, &beta, &delta_value);
        let gc = vc_c.update_commitment(gc, beta[0], delta_value[0]);
        let gc = vc_c.update_commitment(gc, beta[1], delta_value[1]);

        assert!(gc == gcd);
        // Verify that the updated witnesses are correct
        for i in 0..alpha.len() {
            assert!(vc_c.verify(&vc_p, gc, alpha[i], v[alpha[i]], updated_gq[i]));
        }
    }
}
