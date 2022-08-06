use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::prelude::SmallRng;
use ndarray_rand::rand_distr::Uniform;

use std::io::Write;
use std::io::BufRead;

use crate::expected_pi;

type V = f32;

pub struct VectorModel {
    pub freqs: Array<u64, Ix1>,
    pub code: Array<u8, Ix2>,
    pub path: Array<u32, Ix2>,
    pub in_vecs: Array<V, Ix3>,
    pub out_vecs: Array<V, Ix2>,
    pub alpha: f64,
    pub counts: Array<f32, Ix2>,
}

impl VectorModel {
    pub fn new(dim: usize, nsenses: usize, lexsize: usize, codelen: usize, alpha: f64, rng: &mut SmallRng) -> VectorModel {
        let u = Uniform::new(-0.5f32/dim as f32, 0.5f32/dim as f32);
        VectorModel {
            freqs: Array::zeros(lexsize),
            code: Array::from_elem((lexsize, codelen), u8::MAX),
            path: Array::zeros((lexsize, codelen)),
            in_vecs: Array::random_using((lexsize, nsenses, dim), u, rng),
            out_vecs: Array::random_using((lexsize, dim), u, rng),
            alpha: alpha,
            counts: Array::zeros((lexsize, nsenses)),
        }
    }

    pub fn load_model(path: &str) -> Result<VectorModel, Box<dyn std::error::Error>> {
        let mut rf = std::io::BufReader::new(
            std::fs::File::open(path.to_string())?
        );
    
        let mut line = String::new();

        rf.read_line(&mut line)?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() != 3 {
            return Err("bad header format on line 1".into());
        }

        let lexsize = parts[0].parse::<usize>()?;
        let dim = parts[1].parse::<usize>()?;
        let nsenses = parts[2].parse::<usize>()?;

        rf.read_line(&mut line)?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err("bad header format on line 2".into());
        }

        let alpha = parts[0].parse::<f64>()?;
        let _d = parts[1].parse::<f64>()?;

        rf.read_line(&mut line)?;
        let codelen = line.parse::<usize>()?;

        let vm = VectorModel{
            freqs: Array::zeros(lexsize),
            code: Array::from_elem((lexsize, codelen), u8::MAX),
            path: Array::zeros((lexsize, codelen)),
            in_vecs: Array::zeros((lexsize, nsenses, dim)),
            out_vecs: Array::zeros((lexsize, dim)),
            alpha,
            counts: Array::zeros((lexsize, nsenses)),
        };

        Ok(vm)
    }

    pub fn save_model<F>(path: &str, vm: &VectorModel, min_prob: f64, id2word: F)
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
    }

    pub fn nmeanings(&self) -> usize { self.in_vecs.shape()[1] }
}
