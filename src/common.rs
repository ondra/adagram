use ndarray::Array;
use ndarray::prelude::*;
use ndarray::{Data, DataMut};

use crate::simd;
use spfunc::gamma::digamma;
use std::sync::OnceLock;

type F = f64;

// fn digamma(x: F) -> F { x.ln() - 0.5*x }
// ^ bad, maybe use https://math.stackexchange.com/a/1446110

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

pub fn mean_beta(a: F, b: F) -> F {
    a / (a + b)
}
pub fn meanlog_beta(a: F, b: F) -> F {
    digamma(a) - digamma(a + b)
}
//fn mean_mirror(a: F, b: F) -> F { mean_beta(b, a) }
pub fn meanlog_mirror(a: F, b: F) -> F {
    meanlog_beta(b, a)
}

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
    let t = in_vecs.len_of(Axis(1));
    let x = x as usize;
    let y = y as usize;

    let codes = codes.index_axis(Axis(0), y);
    let paths = paths.index_axis(Axis(0), y);
    let codes_slice = codes
        .as_slice()
        .expect("expected contiguous codes storage");
    let paths_slice = paths
        .as_slice()
        .expect("expected contiguous paths storage");

    let in_vecs_x = in_vecs.index_axis(Axis(0), x);
    let in_vecs_x_slice = in_vecs_x
        .as_slice()
        .expect("expected contiguous in_vec storage");
    let dim = in_vecs_x.len_of(Axis(1));
    debug_assert_eq!(in_vecs_x_slice.len(), in_vecs_x.len_of(Axis(0)) * dim);

    let z_slice = z.as_slice_mut().expect("expected contiguous z storage");
    debug_assert!(z_slice.len() >= t);

    for i in 0..codes_slice.len() {
        let code = codes_slice[i];
        if code == u8::MAX {
            break;
        }

        let out_vec = out_vecs.index_axis(Axis(0), paths_slice[i] as usize);
        let sign = 1.0f32 - 2.0f32 * (code as f32);
        let out_vec_slice = out_vec
            .as_slice()
            .expect("expected contiguous out_vec storage");

        for k in 0..t {
            let in_slice = &in_vecs_x_slice[k * dim..(k + 1) * dim];
            let f = simd::dot_f32(in_slice, out_vec_slice);
            z_slice[k] += logsigmoid_f32(f * sign) as f64;
        }
    }
}

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
    if simd::SIMD_ASSUME_ALIGNED_SLICES {
        std::thread_local! {
            static CHECKED_GRAD_ALIGNMENT: std::cell::Cell<bool> = std::cell::Cell::new(false);
        }
        CHECKED_GRAD_ALIGNMENT.with(|checked| {
            if !checked.get() {
                let dim = out_grad.len();
                simd::assert_simd_preconditions(dim, out_grad.as_ptr(), "out_grad");
                simd::assert_simd_preconditions(dim, in_grad.as_ptr(), "in_grad");
                checked.set(true);
            }
        });
    }

    let mut pr = 0.;
    let t = counts.len_of(Axis(1));
    let x = x as usize;
    let _y = y as usize;

    let in_grad_slice = in_grad
        .as_slice_mut()
        .expect("expected contiguous in_grad storage");
    in_grad_slice.fill(0.0);

    {
        let in_vecs_view = in_vecs.view();
        let in_vecs_x = in_vecs_view.index_axis(Axis(0), x);
        let in_vecs_x_slice = in_vecs_x
            .as_slice()
            .expect("expected contiguous in_vec storage");
        let dim = in_vecs_x.len_of(Axis(1));
        debug_assert_eq!(in_vecs_x_slice.len(), in_vecs_x.len_of(Axis(0)) * dim);
        debug_assert_eq!(in_grad_slice.len(), t * dim);
        let z_slice = z.as_slice().expect("expected contiguous z storage");
        debug_assert!(z_slice.len() >= t);

        let codes = codes.index_axis(Axis(0), y as usize);
        let paths = paths.index_axis(Axis(0), y as usize);
        let codes_slice = codes
            .as_slice()
            .expect("expected contiguous codes storage");
        let paths_slice = paths
            .as_slice()
            .expect("expected contiguous paths storage");

        for i in 0..codes_slice.len() {
            let code = codes_slice[i];
            if code == u8::MAX {
                break;
            }

            let code_f = code as f64;
            let sign = 1.0f32 - 2.0f32 * (code as f32);

            // let mut out_vec = vm.out_vecs.slice_mut(s![path, ..]);
            let mut out_vec = out_vecs.index_axis_mut(Axis(0), paths_slice[i] as usize);

            let out_grad_slice = out_grad
                .as_slice_mut()
                .expect("expected contiguous out_grad storage");
            out_grad_slice.fill(0.0);

            {
                let out_vec_ro = out_vec.view();
                let out_slice = out_vec_ro
                    .as_slice()
                    .expect("expected contiguous out_vec storage");
                debug_assert_eq!(out_slice.len(), dim);
                for k in 0..t {
                    if z_slice[k] < sense_threshold {
                        continue;
                    }
                    let in_slice = &in_vecs_x_slice[k * dim..(k + 1) * dim];
                    let f = simd::dot_f32(in_slice, out_slice);

                    if compute_ll {
                        pr += z_slice[k] * (logsigmoid_f32(f * sign) as f64);
                    }

                    let d = (1. - code_f) - (sigmoid_f32(f) as f64);
                    let g = (z_slice[k] * lr * d) as f32;

                    {
                        let in_grad_row = &mut in_grad_slice[k * dim..(k + 1) * dim];
                        simd::axpy_f32(in_grad_row, g, out_slice);
                    }

                    {
                        simd::axpy_f32(out_grad_slice, g, in_slice);
                    }
                }
            }

            let out_vec_slice = out_vec
                .as_slice_mut()
                .expect("expected contiguous out_vec storage");
            simd::axpy_f32(out_vec_slice, 1.0, out_grad_slice);
        }
    }

    let mut in_vecs_x = in_vecs.index_axis_mut(Axis(0), x);
    let dim = in_vecs_x.len_of(Axis(1));
    let rows = in_vecs_x.len_of(Axis(0));
    let in_vecs_x_slice = in_vecs_x
        .as_slice_mut()
        .expect("expected contiguous in_vec storage");
    debug_assert_eq!(in_vecs_x_slice.len(), rows * dim);
    debug_assert_eq!(in_grad_slice.len(), t * dim);
    let z_slice = z.as_slice().expect("expected contiguous z storage");
    for k in 0..t {
        if z_slice[k] < sense_threshold {
            continue;
        }
        let in_vec_slice = &mut in_vecs_x_slice[k * dim..(k + 1) * dim];
        let in_grad_row = &in_grad_slice[k * dim..(k + 1) * dim];
        simd::axpy_f32(in_vec_slice, 1.0, in_grad_row);
    }
    pr
}

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
    for k in 0..counts.len_of(Axis(1)) {
        counts[[x as usize, k]] += (lr2
            * (local_counts[[k]] * freqs[[x as usize]] as f64 - counts[[x as usize, k]] as f64))
            as f32;
    }
}

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

fn _cast_slice<T>(s: &[T]) -> &[u8] {
    let nbytes = std::mem::size_of_val(s);
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, nbytes) }
}

/*
fn load_dict(path: &str) -> Result<(Vec<u64>, Vec<String>), Box<dyn std::error::Error>> {
    let br = std::io::BufReader::new(
        std::fs::File::open(path.to_string())?
    );

    let vr: Result<Vec<(u64, usize, String)>, Box<dyn std::error::Error>>
            = br.lines().enumerate().map(|(lineno, line)| {
        let l = line.unwrap();
        let parts: Vec<_> = l.split_whitespace().collect();
        if parts.len() != 2 {
            return Err("too many whitespace separators in dictionary".into());
        }
        let word = parts[0];
        let frq = parts[1].parse::<u64>()?;
        Ok((frq, lineno, word.to_string()))
    }).collect();
    let mut v = vr?;

    v.sort();
    v.reverse();
    //v.sort_by_key(|w| Reverse(*w));

    let (freqs, ix2word): (Vec<u64>, Vec<String>) = v.into_iter().map(|(f, _, w)| (f, w)).unzip();
    Ok((freqs, ix2word))
}

fn save_model<F>(path: &str, vm: &VectorModel, min_prob: f64, id2word: F)
        -> Result<(), Box<dyn std::error::Error>>
        where F: Fn(u32) -> String{
    let mut vecf = std::io::BufWriter::new(
        std::fs::File::create(path.to_string() + ".txt")?);

    let s = vm.in_vecs.shape();
    write!(vecf, "{} {} {}\n", s[0], s[2], s[1])?;
    write!(vecf, "{} {}\n", vm.alpha, 0)?;
    write!(vecf, "{}\n", vm.code.len_of(Axis(1)))?;

    for &e in vm.freqs.iter() { vecf.write(&e.to_le_bytes())?; };
    for &e in vm.code.iter() { vecf.write(&e.to_le_bytes())?; };
    for &e in vm.path.iter() { vecf.write(&e.to_le_bytes())?; };
    for &e in vm.counts.iter() { vecf.write(&e.to_le_bytes())?; };
    for &e in vm.out_vecs.iter() { vecf.write(&e.to_le_bytes())?; };
    //write!(vecf, "\n")?;

    let mut z = Array::<f64, Ix1>::zeros(s[1]);

    for v in 0..s[0] {
        let nsenses = expected_pi(&vm, v as u32, &mut z);
        write!(vecf, "{}\n", id2word(v as u32))?;
        write!(vecf, "{}\n", nsenses)?;
        for k in 0..s[1] {
            if z[k] < min_prob { continue; }
            write!(vecf, "{}\n", k+1)?;
            for e in vm.in_vecs.slice(s![v, k, ..]).iter() {
                vecf.write(&e.to_le_bytes())?;
            };
            write!(vecf, "\n")?;
        }
    }

    vecf.flush()?;
    std::mem::drop(vecf);

    Ok(())
}*/
