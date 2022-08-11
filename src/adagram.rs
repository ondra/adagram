use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::prelude::SmallRng;
use ndarray_rand::rand_distr::Uniform;

use std::io::Write;
use std::io::Read;
use std::io::BufRead;

use crate::common::expected_pi;

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

    pub fn load_model(path: &str) -> Result<(VectorModel, Vec<String>), Box<dyn std::error::Error>> {
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

        line.clear();
        rf.read_line(&mut line)?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err("bad header format on line 2".into());
        }

        let alpha = parts[0].parse::<f64>()?;
        let _d = parts[1].parse::<f64>()?;

        line.clear();
        rf.read_line(&mut line)?;
        let codelen = line.trim().parse::<usize>()?;

        let mut vm = VectorModel{
            freqs: Array::zeros(lexsize),
            code: Array::from_elem((lexsize, codelen), u8::MAX),
            path: Array::zeros((lexsize, codelen)),
            in_vecs: Array::zeros((lexsize, nsenses, dim)),
            out_vecs: Array::zeros((lexsize, dim)),
            alpha,
            counts: Array::zeros((lexsize, nsenses)),
        };
        
        let mut buf = [0u8; 8];
        for i in 0..lexsize { 
            rf.read_exact(&mut buf)?;
            vm.freqs[i] = u64::from_le_bytes(buf);
        }

        let mut buf = [0u8; 1];
        for i in 0..lexsize {
            for j in 0..codelen {
                rf.read_exact(&mut buf)?;
                vm.code[[i, j]] = u8::from_le_bytes(buf);
            }
        }

        let mut buf = [0u8; 4];
        for i in 0..lexsize {
            for j in 0..codelen {
                rf.read_exact(&mut buf)?;
                vm.path[[i, j]] = u32::from_le_bytes(buf);
            }
        }

        let mut buf = [0u8; 4];
        for i in 0..lexsize {
            for j in 0..nsenses {
                rf.read_exact(&mut buf)?;
                vm.counts[[i, j]] = f32::from_le_bytes(buf);
            }
        }
        
        let mut buf = [0u8; 4];
        for i in 0..lexsize {
            for j in 0..dim {
                rf.read_exact(&mut buf)?;
                vm.out_vecs[[i, j]] = f32::from_le_bytes(buf);
            }
        }
        
        let mut id2str = Vec::<String>::new();

        let mut line = String::new();
        let mut buf = [0u8; 4];
        for i in 0..lexsize {
            line.clear();
            rf.read_line(&mut line)?;
            let word = line.trim().to_string();
            id2str.push(word);

            line.clear();
            rf.read_line(&mut line)?;
            let ns = line.trim().parse::<usize>()?;
            for _j in 0..ns {
                line.clear();
                rf.read_line(&mut line)?;
                let s = line.trim().parse::<usize>()? - 1;
                for k in 0..dim {
                    rf.read_exact(&mut buf)?;
                    vm.in_vecs[[i, s, k]] = f32::from_le_bytes(buf);
                }
                line.clear();
                rf.read_line(&mut line)?;
                assert!(line.trim() == "");
            }
        }
        
        Ok((vm, id2str))
    }

    pub fn save_model<F>(&self, path: &str, min_prob: f64, id2word: F)
            -> Result<(), Box<dyn std::error::Error>>
            where F: Fn(u32) -> String{
        let mut vecf = std::io::BufWriter::new(
            std::fs::File::create(path.to_string())?);
    
        let s = self.in_vecs.shape();
        write!(vecf, "{} {} {}\n", s[0], s[2], s[1])?;
        write!(vecf, "{} {}\n", self.alpha, 0)?;
        write!(vecf, "{}\n", self.code.len_of(Axis(1)))?;

        for &e in self.freqs.iter() { vecf.write(&e.to_le_bytes())?; };
        for &e in self.code.iter() { vecf.write(&e.to_le_bytes())?; };
        for &e in self.path.iter() { vecf.write(&e.to_le_bytes())?; };
        for &e in self.counts.iter() { vecf.write(&e.to_le_bytes())?; };
        for &e in self.out_vecs.iter() { vecf.write(&e.to_le_bytes())?; };
        //write!(vecf, "\n")?;

        let mut z = Array::<f64, Ix1>::zeros(s[1]);

        for v in 0..s[0] {
            let nsenses = expected_pi(&self, v as u32, &mut z);
            write!(vecf, "{}\n", id2word(v as u32))?;
            write!(vecf, "{}\n", nsenses)?;
            for k in 0..s[1] {
                if z[k] < min_prob { continue; }
                write!(vecf, "{}\n", k+1)?;
                for e in self.in_vecs.slice(s![v, k, ..]).iter() {
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
