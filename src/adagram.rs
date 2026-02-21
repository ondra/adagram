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

    pub normed: bool,
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
            alpha,
            counts: Array::zeros((lexsize, nsenses)),
            normed: false,
        }
    }

    pub fn load_model(path: &str) -> Result<(VectorModel, Vec<String>), Box<dyn std::error::Error>> {
        let mut rf = std::io::BufReader::new(
            std::fs::File::open(path)?
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
            normed: false,
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
        rf.read_line(&mut line)?;
        let mut buf = [0u8; 4];

        // see the description below, but should not be necessary unless using legacy data
        const ALLOW_MALFORMED_MODELS: bool = false;

        for i in 0..lexsize {
            let word = line.trim().to_string();
            id2str.push(word);

            line.clear();
            rf.read_line(&mut line)?;
            let hw_nsenses = line.trim().parse::<usize>().map_err(|_| "bad number of senses")?;

            line.clear();
            rf.read_line(&mut line)?;
            let mut jj = 0;
            while jj < nsenses {
                let senseno = line.trim().parse::<usize>().map_err(|_| "bad sense number")? - 1;
                if senseno > nsenses {
                    return Err(format!("bad senseno {}", senseno).into());
                }
                for k in 0..dim {
                    rf.read_exact(&mut buf)?;
                    vm.in_vecs[[i, senseno, k]] = f32::from_le_bytes(buf);
                }
                line.clear();
                rf.read_line(&mut line)?;
                if line.trim() != "" {
                    return Err(format!("trailing characters after binary vector").into());
                };

                jj += 1;

                if !ALLOW_MALFORMED_MODELS {
                    line.clear();
                    rf.read_line(&mut line)?;
                    if jj == hw_nsenses { break }
                } else {
                ///////////////// HACK HACK HACK ////////////////////////
                //  The original writer does not store the number of senses
                //  properly; the senses are stored only when their probability
                //  crosses a threshold, but this threshold is not the same
                //  as the threshold used to calculate the number of senses,
                //  so the encoded sense count might differ from the amount
                //  of actually stored senses if a non-default minimum probability
                //  is used.
                //
                //  This code tries to guess whether to read a new sense or
                //  a whole new record by a heuristic lookahead.
                //
                //  Note that this code is safe when the values are lemposes
                //  in the format 'word-x'.
                ///////////////// HACK HACK HACK ////////////////////////
                const VERBOSE: bool = false;
                rf.read_line(&mut line)?;
                if line.is_empty() { break; }
                let s1 = line.trim();
                match s1.parse::<usize>() {
                    Ok(n) => /* it's a number */ {
                        if (n-1) > senseno && (n-1) < nsenses {
                            /* looks like a new sense */
                            let buf = rf.fill_buf()?;
                            let mut state = 0;  // 0 init
                            for ix in 0..buf.len() {
                                let cr = buf[ix];
                                let next_state = match state {
                                    0 => if cr >= b'1' && cr <= b'9' {        1 } else { 99 },
                                    1 => if cr >= b'0' && cr <= b'2' {        2 }
                                        else if cr == 10 {                    3 }else { 99 },
                                    2 => if cr == 10 {                        3 } else { 99 },
                                    3 => if cr >= b'1' && cr <= b'9' {        4 } else { 99 },
                                    4 => if cr >= b'0' && cr <= b'2'  {       5 }
                                         else if cr == 10 {                   10} else { 99 },
                                    5 => if cr == 10 {                        10} else { 99 },
                                    _ =>                                      99,
                                };
                                state = next_state;
                                if state == 10 || state == 99 { break; }
                            }
                            match state {
                                10 => {
                                    if VERBOSE { eprintln!("parsing record after id {}, got {}", i, state); }
                                    break; // got two numbers followed by newlines
                                },
                                99 => {
                                    // not a new word
                                },
                                _ => {
                                    if VERBOSE { eprintln!("parsing record after id {}, got {}", i, state); }
                                },
                            }

                            // this is probably a new sense
                            // continue;
                        } else {
                            /* it's a number, but the value is unexpected -> a new lexicon element */
                            break;
                        }
                    },
                    Err(_) => /* it's surely a new word */ break,
                }
                /////////// END HACK /////////////////
                /////////// END HACK /////////////////
                }
            }
        }
        
        Ok((vm, id2str))
    }

    pub fn save_model<F>(&self, vecf: &mut std::fs::File, min_prob: f64, id2word: F, extra: String)
            -> Result<(), Box<dyn std::error::Error>>
            where F: Fn(u32) -> String {
        let mut vecwr = std::io::BufWriter::new(vecf);
        let s = self.in_vecs.shape();
        writeln!(vecwr, "{} {} {}", s[0], s[2], s[1])?;
        writeln!(vecwr, "{} {}", self.alpha, 0)?;
        writeln!(vecwr, "{}", self.code.len_of(Axis(1)))?;

        for &e in self.freqs.iter() { vecwr.write_all(&e.to_le_bytes())?; };
        for &e in self.code.iter() { vecwr.write_all(&e.to_le_bytes())?; };
        for &e in self.path.iter() { vecwr.write_all(&e.to_le_bytes())?; };
        for &e in self.counts.iter() { vecwr.write_all(&e.to_le_bytes())?; };
        for &e in self.out_vecs.iter() { vecwr.write_all(&e.to_le_bytes())?; };

        let mut z = Array::<f64, Ix1>::zeros(s[1]);

        for v in 0..s[0] {
            let nsenses = expected_pi(&self.counts, self.alpha, v as u32, &mut z, min_prob);
            writeln!(vecwr, "{}", id2word(v as u32))?;
            writeln!(vecwr, "{}", nsenses)?;
            for k in 0..s[1] {
                if z[k] < min_prob { continue; }
                writeln!(vecwr, "{}", k+1)?;
                for e in self.in_vecs.slice(s![v, k, ..]).iter() {
                    vecwr.write_all(&e.to_le_bytes())?;
                };
                writeln!(vecwr)?;
            }
        }
        vecwr.flush()?;
        writeln!(vecwr, "extrainfo: {}", extra)?;
        vecwr.flush()?;
        Ok(())
    }

    pub fn save_model_atomic<F>(&self, path: &str, min_prob: f64, id2word: F, extra: String)
            -> Result<(), Box<dyn std::error::Error>>
            where F: Fn(u32) -> String {
        let tmppath = path.to_string() + ".tmp";
        let mut vecf = std::fs::File::create(&tmppath)?;
    
        self.save_model(&mut vecf, min_prob, id2word, extra)?;
        std::mem::drop(vecf);
        std::fs::rename(tmppath, path)?;

        Ok(())
    }

    pub fn nmeanings(&self) -> usize { self.in_vecs.shape()[1] }
    pub fn newz(&self) -> Array::<f64, Ix1> {
        Array::<f64, Ix1>::zeros(self.nmeanings())
    }
    pub fn var_update_z(&self, head_id: u32, ctx_id: u32, z: &mut Array::<f64, Ix1>) {
        crate::common::var_update_z(&self.in_vecs, &self.out_vecs, &self.code, &self.path, head_id, ctx_id, z);
    }
    pub fn expected_pi(&self, head_id: u32, z: &mut Array::<f64, Ix1>, min_prob: f64, uniform_prob: bool) -> u32 {
        let n_senses = expected_pi(&self.counts, self.alpha, head_id, z, min_prob);

        if uniform_prob {
            for zk in z.iter_mut() {
                if *zk < min_prob { *zk = 0.; }
                else { *zk = 1. / n_senses as f64; }
            }
        } else {
            for zk in z.iter_mut() {
                if *zk < min_prob { *zk = 0.; }
                *zk = zk.ln();  // ???
            }
        }
        n_senses
    }
}

pub fn parse_window(window: Option<usize>, modelpath: &str) -> Option<usize> {
    window.or(modelpath.split('.').find_map(|part| {
        part.strip_prefix('w')?
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0 && n < 100)
    }))
}
