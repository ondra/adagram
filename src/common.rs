use ndarray::Array;
use ndarray::{Data,DataMut};
use ndarray::prelude::*;

use crate::adagram::VectorModel;
use spfunc::gamma::digamma;

type F = f64;

// fn digamma(x: F) -> F { x.ln() - 0.5*x }
// ^ bad, maybe use https://math.stackexchange.com/a/1446110

pub fn mean_beta(a: F, b: F) -> F { a / (a + b) }
pub fn meanlog_beta(a: F, b: F) -> F { digamma(a) - digamma(a + b) }
//fn mean_mirror(a: F, b: F) -> F { mean_beta(b, a) }
pub fn meanlog_mirror(a: F, b: F) -> F { meanlog_beta(b, a) }


pub fn expected_pi<C: Data<Elem=f32>, Z: DataMut<Elem=f64>>
        (counts: &ArrayBase<C, Ix2>, alpha: f64, w: u32, pi: &mut ArrayBase<Z, Ix1>) -> u32 {
    let min_prob = 1e-3f64;  // fixme

    let nmeanings = counts.len_of(Axis(1));

    let mut r = 1.;
    let mut senses = 0;
    let mut ts = counts.slice(s![w as usize, ..]).sum() as f64;
    
    for k in 0..nmeanings-1 {
        ts = f64::max(ts - counts[[w as usize, k]] as f64, 0.);
        let a = 1. + counts[[w as usize, k]] as f64;
        let b = alpha + ts;
        pi[k] = mean_beta(a, b) * r;
        if pi[k] >= min_prob { senses += 1; }
        r = f64::max(r - pi[k], 0.)
    }
    pi[nmeanings-1] = r; 
    if r >= min_prob { senses += 1; }
    senses
}

pub fn var_init_z<C: Data<Elem=f32>, Z: DataMut<Elem=f64>>
        (counts: &ArrayBase<C, Ix2>, alpha: f64, w: u32, pi: &mut ArrayBase<Z, Ix1>) -> u32 {
    let min_prob = 1e-3f64;

    let mut r = 0f64;
    let mut x = 1f64;
    let mut senses = 0u32;
    for i in 0..pi.len() {
        pi[i] = counts[[w as usize, i]] as f64;
    }
    let mut ts = pi.sum();
    for k in 0..pi.len()-1 {
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
    pi[lp-1] = r; 
    if x >= min_prob { senses += 1; }
    senses
}

pub fn sigmoid   (x: f64) -> f64 { 1. / (1.+(-x).exp()) }
pub fn logsigmoid(x: f64) -> f64 {     -(1.+(-x).exp()).ln() }

pub fn var_update_z<
    I: Data<Elem=f32>, O: Data<Elem=f32>,
    C: Data<Elem=u8>, P: Data<Elem=u32>>
        (in_vecs: &ArrayBase<I, Ix3>, out_vecs: &ArrayBase<O, Ix2>,
         codes: &ArrayBase<C, Ix2>, paths: &ArrayBase<P, Ix2>,
         x: u32, y: u32, z: &mut Array<f64, Ix1>) {
    let t = in_vecs.len_of(Axis(1));
    let x = x as usize;
    let y = y as usize;
    
    let codes = codes.index_axis(Axis(0), y as usize);
    let paths = paths.index_axis(Axis(0), y as usize);
    for (code, path) in std::iter::zip(codes, paths) {
        if *code == u8::MAX { break; }

        let out_vec = out_vecs.slice(s![*path as usize, ..]);

        for k in 0..t {
            let in_vec = in_vecs.slice(s![x, k, ..]);
            let f = in_vec.dot(&out_vec) as f64;
            z[k] += logsigmoid(f * (1. - 2.*(*code as f64)));
        }
    }
}

fn _skip_gram(vm: &mut VectorModel, in_vec: &Array<f32, Ix1>, x: u32) -> f64 {
    let x = x as usize;
    let mut pr = 0.;
    
    for n in 0..vm.code.len_of(Axis(1)) {
        let code = vm.code[[x, n]];
        let path = vm.path[[x, n]] as usize;
        if code == u8::MAX { break; }

        let out_vec = vm.out_vecs.slice(s![path, ..]);
        let f = in_vec.dot(&out_vec) as f64;

        pr += logsigmoid(f * (1. - 2.*(code as f64)));
    }
    pr
}

pub fn in_place_update<
    I: ndarray::DataMut<Elem = f32>,
    O: ndarray::DataMut<Elem = f32>,
    D: ndarray::DataMut<Elem = f32>,
    Z: ndarray::Data<Elem = f64>,
    Cc: ndarray::Data<Elem = f32>,

    C: ndarray::Data<Elem = u8>,
    P: ndarray::Data<Elem = u32>,
    >
    (in_vecs: &mut ArrayBase<I, Ix3>,
     out_vecs: &mut ArrayBase<O, Ix2>,
     counts: &ArrayBase<Cc, Ix2>,
     x: u32, y: u32, z: &ArrayBase<Z, Ix1>,
        codes: &ArrayBase<C, Ix2>, paths: &ArrayBase<P, Ix2>,
                   lr: f64,
                   in_grad: &mut ArrayBase<D, Ix2>, out_grad: &mut ArrayBase<D, Ix1>,
                   sense_threshold: f64) -> f64 {
    let mut pr = 0.;
    let t = counts.len_of(Axis(1));
    let m = in_vecs.len_of(Axis(2));
    let x = x as usize;
    let _y = y as usize;

    in_grad.fill(0.);    

    let codes = codes.index_axis(Axis(0), y as usize);
    let paths = paths.index_axis(Axis(0), y as usize);
    for (code, path) in std::iter::zip(codes, paths) {
        if *code == u8::MAX { break; }

        //let mut out_vec = vm.out_vecs.slice_mut(s![path, ..]);
        let mut out_vec = out_vecs.index_axis_mut(Axis(0), *path as usize);

        out_grad.fill(0.);

        for k in 0..t {
            if z[k] < sense_threshold { continue; }
            let in_vec = in_vecs.slice(s![x, k, ..]);
            let f = out_vec.dot(&in_vec) as f64;
            
            pr += z[k] * logsigmoid(f * (1. - 2.*(*code as f64)));
            let d = 1. - *code as f64 - sigmoid(f);
            let g = z[k] * lr * d;

            for i in 0..m {
                in_grad[[k, i]] += (g * out_vec[i] as f64) as f32;
                out_grad[i] += (g * in_vec[i] as f64) as f32;
            }
        }

        for i in 0..m { out_vec[i] += out_grad[i]; }
    }

    for k in 0..t {
        if z[k] < sense_threshold { continue; }
        let mut in_vec = in_vecs.slice_mut(s![x, k, ..]);
        //let mut in_vecc = vm.in_vecs.index_axis_mut(Axis(0), x);
        //let mut in_vec = in_vecc.index_axis_mut(Axis(0), k);
        in_vec += &in_grad.slice(s![k, ..]);
    }
    pr
}

pub fn var_update_counts<
        A: ndarray::Data<Elem = u64>,
        B: ndarray::DataMut<Elem = f32>,
        C: ndarray::Data<Elem = f64>
>(
    freqs: &ArrayBase<A, Ix1>, counts: &mut ArrayBase<B, Ix2>, x: u32, local_counts: &ArrayBase<C, Ix1>, lr2: f64)
{
    for k in 0..counts.len_of(Axis(1)) {
        counts[[x as usize, k]] += (lr2 *
            (local_counts[[k as usize]] * freqs[[x as usize]] as f64 - counts[[x as usize, k]] as f64)) as f32;
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
        *e = *e / sum_x;
    }
}

fn _cast_slice<T>(s: &[T]) -> &[u8] {
    let nbytes = s.len() * std::mem::size_of::<T>();
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




