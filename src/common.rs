use ndarray::Array;
use ndarray::prelude::*;
use ndarray::{Data, DataMut};
use ndarray::Dimension;

use crate::simd;
use spfunc::gamma::digamma;
use std::sync::OnceLock;

use multiversion::multiversion;

/// Check that a pointer and dimension satisfy SIMD alignment requirements.
/// Panics if `SIMD_ASSUME_ALIGNED_SLICES` is enabled and the preconditions are violated.
#[inline(always)]
pub fn assert_simd_aligned(dim_padded: usize, base_ptr: *const f32, what: &str) {
    simd::assert_simd_preconditions(dim_padded, base_ptr, what);
}

type F = f64;

/// Controls whether `sigmoid`/`logsigmoid` use precise `exp/log` math (slower) or a lookup-table
/// approximation (faster). This is a compile-time choice on purpose to keep the hot loops free of
/// runtime conditionals.
pub const PRECISE_MATH: bool = false;

const FAST_MATH_MAX_X: f32 = 8.0;
const FAST_MATH_N: usize = 4096;

struct FastMathTable {
    inv_step: f32,
    sigmoid: Vec<f32>,
    logsigmoid: Vec<f32>,
}

impl FastMathTable {
    fn new() -> Self {
        let step = (2.0 * FAST_MATH_MAX_X) / (FAST_MATH_N as f32 - 1.0);
        let inv_step = 1.0 / step;
        let mut sigmoid = Vec::with_capacity(FAST_MATH_N);
        let mut logsigmoid = Vec::with_capacity(FAST_MATH_N);
        for i in 0..FAST_MATH_N {
            let x = -FAST_MATH_MAX_X + (i as f32) * step;
            let s = 1.0 / (1.0 + (-x).exp());
            sigmoid.push(s);
            logsigmoid.push(s.ln());
        }
        Self {
            inv_step,
            sigmoid,
            logsigmoid,
        }
    }

    #[inline(always)]
    fn idx(&self, x: f32) -> (usize, f32) {
        let t = (x + FAST_MATH_MAX_X) * self.inv_step;
        let mut i = t as usize;
        if i + 1 >= FAST_MATH_N {
            i = FAST_MATH_N - 2;
        }
        let frac = t - i as f32;
        (i, frac)
    }
}

static FAST_MATH: OnceLock<FastMathTable> = OnceLock::new();

#[inline(always)]
fn fast_math_table() -> &'static FastMathTable {
    if let Some(t) = FAST_MATH.get() {
        t
    } else {
        FAST_MATH.get_or_init(FastMathTable::new)
    }
}

#[inline(always)]
fn contig_slice<'a, S, D, T>(a: &'a ndarray::ArrayBase<S, D>, what: &'static str) -> &'a [T]
where
    S: Data<Elem = T>,
    D: Dimension,
{
    a.as_slice()
        .unwrap_or_else(|| panic!("expected contiguous {what} storage"))
}

#[inline(always)]
fn contig_slice_mut<'a, S, D, T>(
    a: &'a mut ndarray::ArrayBase<S, D>,
    what: &'static str,
) -> &'a mut [T]
where
    S: DataMut<Elem = T>,
    D: Dimension,
{
    a.as_slice_mut()
        .unwrap_or_else(|| panic!("expected contiguous {what} storage"))
}

pub fn init_math() {
    if !PRECISE_MATH {
        let _ = fast_math_table();
    }
}

#[inline(always)]
fn sigmoid_precise_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[inline(always)]
fn logsigmoid_precise_f32(x: f32) -> f32 {
    if x >= 0.0 {
        -((-x).exp()).ln_1p()
    } else {
        x - (x.exp()).ln_1p()
    }
}

#[inline(always)]
fn sigmoid_fast_f32(table: &FastMathTable, x: f32) -> f32 {
    if x <= -FAST_MATH_MAX_X {
        return 0.0;
    }
    if x >= FAST_MATH_MAX_X {
        return 1.0;
    }
    let (i, frac) = table.idx(x);
    let s0 = table.sigmoid[i];
    let s1 = table.sigmoid[i + 1];
    s0 + (s1 - s0) * frac
}

#[inline(always)]
fn logsigmoid_fast_f32(table: &FastMathTable, x: f32) -> f32 {
    if x <= -FAST_MATH_MAX_X {
        return x;
    }
    if x >= FAST_MATH_MAX_X {
        return 0.0;
    }
    let (i, frac) = table.idx(x);
    let l0 = table.logsigmoid[i];
    let l1 = table.logsigmoid[i + 1];
    l0 + (l1 - l0) * frac
}

#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    if PRECISE_MATH {
        sigmoid_precise_f32(x)
    } else {
        sigmoid_fast_f32(fast_math_table(), x)
    }
}

#[inline(always)]
fn logsigmoid_f32(x: f32) -> f32 {
    if PRECISE_MATH {
        logsigmoid_precise_f32(x)
    } else {
        logsigmoid_fast_f32(fast_math_table(), x)
    }
}

#[inline(always)]
fn mean_beta(a: F, b: F) -> F {
    a / (a + b)
}

#[inline(always)]
fn meanlog_beta(a: F, b: F) -> F {
    digamma(a) - digamma(a + b)
}

#[inline(always)]
fn meanlog_mirror(a: F, b: F) -> F {
    meanlog_beta(b, a)
}

#[multiversion(targets = "simd")]
pub fn expected_pi<C: Data<Elem = f32>, Z: DataMut<Elem = f64>>(
    counts: &ArrayBase<C, Ix2>,
    alpha: f64,
    w: u32,
    pi: &mut ArrayBase<Z, Ix1>,
    min_prob: f64,
) -> u32 {
    let nmeanings = counts.len_of(Axis(1));

    let mut r = 1.;
    let mut senses = 0;
    let mut ts = counts.slice(s![w as usize, ..]).sum() as f64;

    for k in 0..nmeanings - 1 {
        ts = f64::max(ts - counts[[w as usize, k]] as f64, 0.);
        let a = 1. + counts[[w as usize, k]] as f64;
        let b = alpha + ts;
        pi[k] = mean_beta(a, b) * r;
        if pi[k] >= min_prob {
            senses += 1;
        }
        r = f64::max(r - pi[k], 0.)
    }
    pi[nmeanings - 1] = r;
    if r >= min_prob {
        senses += 1;
    }
    senses
}

#[multiversion(targets = "simd")]
pub fn var_init_z<C: Data<Elem = f32>, Z: DataMut<Elem = f64>>(
    counts: &ArrayBase<C, Ix2>,
    alpha: f64,
    w: u32,
    pi: &mut ArrayBase<Z, Ix1>,
) -> u32 {
    let min_prob = 1e-3f64;

    let mut r = 0f64;
    let mut x = 1f64;
    let mut senses = 0u32;
    for i in 0..pi.len() {
        pi[i] = counts[[w as usize, i]] as f64;
    }
    let mut ts = pi.sum();
    for k in 0..pi.len() - 1 {
        ts = f64::max(ts - pi[k], 0.);
        let a = 1. + pi[k];
        let b = alpha + ts;
        pi[k] = meanlog_beta(a, b) + r;
        r += meanlog_mirror(a, b);

        let pi_k = mean_beta(a, b) * x;
        x = f64::max(x - pi_k, 0.);
        if pi_k >= min_prob {
            senses += 1;
        }
    }
    let lp = pi.len();
    pi[lp - 1] = r;
    if x >= min_prob {
        senses += 1;
    }
    senses
}

#[multiversion(targets = "simd")]
pub fn var_update_z<
    I: Data<Elem = f32>,
    O: Data<Elem = f32>,
    C: Data<Elem = u8>,
    P: Data<Elem = u32>,
>(
    in_vecs: &ArrayBase<I, Ix3>,
    out_vecs: &ArrayBase<O, Ix2>,
    codes: &ArrayBase<C, Ix2>,
    paths: &ArrayBase<P, Ix2>,
    x: u32,
    y: u32,
    z: &mut Array<f64, Ix1>,
) {
    let x = x as usize;
    let y = y as usize;

    debug_assert!(in_vecs.is_standard_layout());
    debug_assert!(out_vecs.is_standard_layout());
    debug_assert!(codes.is_standard_layout());
    debug_assert!(paths.is_standard_layout());

    let t = in_vecs.len_of(Axis(1));
    let dim = in_vecs.len_of(Axis(2));
    debug_assert_eq!(out_vecs.len_of(Axis(1)), dim);
    debug_assert!(z.len() >= t);

    let codes_all = contig_slice(codes, "codes");
    let paths_all = contig_slice(paths, "paths");
    let codelen = codes.len_of(Axis(1));
    debug_assert_eq!(paths.len_of(Axis(1)), codelen);
    let codes_row = &codes_all[y * codelen..(y + 1) * codelen];
    let paths_row = &paths_all[y * codelen..(y + 1) * codelen];

    let in_all = contig_slice(in_vecs, "in_vecs");
    let x_base = x * t * dim;
    let in_x = &in_all[x_base..x_base + t * dim];

    let out_all = contig_slice(out_vecs, "out_vecs");

    let z_slice = contig_slice_mut(z, "z");

    for i in 0..codes_row.len() {
        let code = codes_row[i];
        if code == u8::MAX {
            break;
        }

        let sign = 1.0f32 - 2.0f32 * (code as f32);
        let path = paths_row[i] as usize;
        let out_slice = &out_all[path * dim..(path + 1) * dim];

        for k in 0..t {
            let in_slice = &in_x[k * dim..(k + 1) * dim];
            let f = simd::dot_f32(in_slice, out_slice);
            z_slice[k] += logsigmoid_f32(f * sign) as f64;
        }
    }
}

#[multiversion(targets = "simd")]
pub fn in_place_update<
    I: ndarray::DataMut<Elem = f32>,
    O: ndarray::DataMut<Elem = f32>,
    D: ndarray::DataMut<Elem = f32>,
    Z: ndarray::Data<Elem = f64>,
    Cc: ndarray::Data<Elem = f32>,
    C: ndarray::Data<Elem = u8>,
    P: ndarray::Data<Elem = u32>,
>(
    in_vecs: &mut ArrayBase<I, Ix3>,
    out_vecs: &mut ArrayBase<O, Ix2>,
    counts: &ArrayBase<Cc, Ix2>,
    x: u32,
    y: u32,
    z: &ArrayBase<Z, Ix1>,
    codes: &ArrayBase<C, Ix2>,
    paths: &ArrayBase<P, Ix2>,
    lr: f64,
    in_grad: &mut ArrayBase<D, Ix2>,
    out_grad: &mut ArrayBase<D, Ix1>,
    sense_threshold: f64,
    compute_ll: bool,
) -> f64 {
    let mut pr = 0.;
    let t = counts.len_of(Axis(1));
    let x = x as usize;

    debug_assert!(in_vecs.is_standard_layout());
    debug_assert!(out_vecs.is_standard_layout());
    debug_assert!(codes.is_standard_layout());
    debug_assert!(paths.is_standard_layout());
    debug_assert!(in_grad.is_standard_layout());
    debug_assert!(out_grad.is_standard_layout());

    let dim = out_vecs.len_of(Axis(1));
    debug_assert_eq!(in_vecs.len_of(Axis(2)), dim);
    debug_assert_eq!(in_vecs.len_of(Axis(1)), t);
    debug_assert_eq!(in_grad.len(), t * dim);
    debug_assert_eq!(out_grad.len(), dim);

    let in_grad_slice = contig_slice_mut(in_grad, "in_grad");
    in_grad_slice.fill(0.0);

    let out_grad_slice = contig_slice_mut(out_grad, "out_grad");
    let z_slice = contig_slice(z, "z");
    debug_assert!(z_slice.len() >= t);

    let codes_all = contig_slice(codes, "codes");
    let paths_all = contig_slice(paths, "paths");
    let codelen = codes.len_of(Axis(1));
    debug_assert_eq!(paths.len_of(Axis(1)), codelen);
    let y = y as usize;
    let codes_row = &codes_all[y * codelen..(y + 1) * codelen];
    let paths_row = &paths_all[y * codelen..(y + 1) * codelen];

    let out_all = contig_slice_mut(out_vecs, "out_vecs");
    let in_all = contig_slice(in_vecs, "in_vecs");
    let x_base = x * t * dim;
    let in_x = &in_all[x_base..x_base + t * dim];

    for i in 0..codes_row.len() {
        let code = codes_row[i];
        if code == u8::MAX {
            break;
        }

        let code_f = code as f64;
        let sign = 1.0f32 - 2.0f32 * (code as f32);

        out_grad_slice.fill(0.0);

        let path = paths_row[i] as usize;
        let out_off = path * dim;
        let out_slice: &[f32] = &out_all[out_off..out_off + dim];

        for k in 0..t {
            let zk = z_slice[k];
            if zk < sense_threshold {
                continue;
            }
            let in_slice = &in_x[k * dim..(k + 1) * dim];
            let f = simd::dot_f32(in_slice, out_slice);

            if compute_ll {
                pr += zk * (logsigmoid_f32(f * sign) as f64);
            }

            let d = (1. - code_f) - (sigmoid_f32(f) as f64);
            let g = (zk * lr * d) as f32;

            let in_grad_row = &mut in_grad_slice[k * dim..(k + 1) * dim];
            simd::axpy_f32(in_grad_row, g, out_slice);
            simd::axpy_f32(out_grad_slice, g, in_slice);
        }

        let out_row = &mut out_all[out_off..out_off + dim];
        simd::axpy_f32(out_row, 1.0, out_grad_slice);
    }

    let in_all_mut = contig_slice_mut(in_vecs, "in_vecs");
    let x_base = x * t * dim;
    let in_x_mut = &mut in_all_mut[x_base..x_base + t * dim];
    for k in 0..t {
        if z_slice[k] < sense_threshold {
            continue;
        }
        let in_vec_slice = &mut in_x_mut[k * dim..(k + 1) * dim];
        let in_grad_row = &in_grad_slice[k * dim..(k + 1) * dim];
        simd::axpy_f32(in_vec_slice, 1.0, in_grad_row);
    }
    pr
}

#[multiversion(targets = "simd")]
pub fn var_update_counts<
    A: ndarray::Data<Elem = u64>,
    B: ndarray::DataMut<Elem = f32>,
    C: ndarray::Data<Elem = f64>,
>(
    freqs: &ArrayBase<A, Ix1>,
    counts: &mut ArrayBase<B, Ix2>,
    x: u32,
    local_counts: &ArrayBase<C, Ix1>,
    lr2: f64,
) {
    let freq = freqs[x as usize] as f64;
    let nmeanings = counts.len_of(Axis(1));
    let counts_row = &mut contig_slice_mut(counts, "counts")
        [x as usize * nmeanings..(x as usize + 1) * nmeanings];
    let z_slice = contig_slice(local_counts, "local_counts");
    for k in 0..nmeanings {
        counts_row[k] += (lr2 * (z_slice[k] * freq - counts_row[k] as f64)) as f32;
    }
}

#[multiversion(targets = "simd")]
pub fn exp_normalize(x: &mut Array<f64, Ix1>) {
    let max_x = x.fold(f64::MIN, |acc, e| if *e > acc { *e } else { acc });
    let mut sum_x = 0f64;
    for e in x.iter_mut() {
        *e = (*e - max_x).exp();
        sum_x += *e;
    }
    for e in x.iter_mut() {
        *e /= sum_x;
    }
}

