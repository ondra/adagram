use clap::Parser;

mod adagram;
mod huffman;

use ndarray::Array;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::SmallRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::distributions;
use ndarray_rand::rand_distr::Distribution;

use crate::adagram::VectorModel;
use std::io::Write;
use std::io::BufRead;
use spfunc::gamma::digamma;

type F = f64;

// fn digamma(x: F) -> F { x.ln() - 0.5*x }
// ^ bad, maybe use https://math.stackexchange.com/a/1446110

fn mean_beta(a: F, b: F) -> F { a / (a + b) }
fn meanlog_beta(a: F, b: F) -> F { digamma(a) - digamma(a + b) }
//fn mean_mirror(a: F, b: F) -> F { mean_beta(b, a) }
fn meanlog_mirror(a: F, b: F) -> F { meanlog_beta(b, a) }

/// Train an adaptive skip-gram model
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    //// training text data
    //train: String,

    //// dictionary file with word frequencies
    //dict: String,

    /// training corpus
    corpname: String,

    /// training attribute
    attrname: String,

    /// window size
    #[clap(long,default_value_t=4)]
    window: u32,

    /// minimum frequency of the word
    #[clap(long,default_value_t=20)]
    min_freq: u64,

    /// remove top K most frequent words
    #[clap(long,default_value_t=0)]
    remove_top_k: u32,

    /// dimensionality of the representations
    #[clap(long,default_value_t=100)]
    dim: usize,

    /// number of word prototypes
    #[clap(long,default_value_t=5)]
    prototypes: usize,

    /// prior probability of allocating a new prototype
    #[clap(long,default_value_t=0.1)]
    alpha: f64,

    /// subsamplVing threshold
    #[clap(long,default_value_t=f32::INFINITY)]
    subsample: f32,

    /// number of epochs to train
    #[clap(long,default_value_t=1)]
    epochs: u32,

    /// randomly reduce size of the context
    #[clap(long)]
    context_cut: bool,

    /// initial weight (count) on first sense for each word
    #[clap(long,default_value_t=-1.)]
    init_count: f32,

    /// minimal probability of a meaning to contribute into gradients
    #[clap(long,default_value_t=1e-10)]
    sense_threshold: f64,

    /// minimal probability of a meaning to save after training
    #[clap(long,default_value_t=1e-3)]
    save_threshold: f64,

    /// minimal probability of a meaning to save after training
    #[clap(long,default_value_t=0.025)]
    start_lr: f64,

    /// path to save the output
    outpath: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let corp = corp::corp::Corpus::open(&args.corpname)?;
    let attr = corp.open_attribute(&args.attrname)?;

    let lexsize = attr.lex.id_range();
    let mut ofreqs: Vec<u64> = vec![0u64; lexsize as usize];
    let mut ixs: Vec<u32> = Vec::new();
    for id in 0..lexsize {
        let freq = attr.rev.count(id);
        ofreqs[id as usize] = freq;
        if freq >= args.min_freq { ixs.push(id); }
    }

    ixs.sort_by_key(|&id| std::cmp::Reverse(&ofreqs[id as usize])); 
    if args.remove_top_k != 0 {
        ixs.drain(0..args.remove_top_k as usize);
    }

    let reduced_lexsize = ixs.len(); 
    println!("pruned lexicon size is {}", reduced_lexsize);

    let mut freqs = vec![0u64; reduced_lexsize];
    let mut oid_to_nid = vec![u32::MAX; lexsize as usize];
    for (nid, oid) in ixs.iter().enumerate() {
        oid_to_nid[*oid as usize] = nid as u32;
        freqs[nid] = ofreqs[*oid as usize];
    }

    println!("building huffman tree");
    let ht = huffman::HuffmanTree::new(&freqs);

    let mut max_codelen = 0;
    for id in 0..reduced_lexsize {
        let (a, _b) = ht.softmax_path(id as u32);
        if a.len() > max_codelen { max_codelen = a.len(); }
    }
    println!("maximum code length is {}", max_codelen);

    let mut rng = SmallRng::seed_from_u64(666);
    let mut vm = VectorModel::new(args.dim, args.prototypes,
        reduced_lexsize, max_codelen, args.alpha, &mut rng);

    vm.freqs.assign(&Array::from_vec(freqs));
    for id in 0..reduced_lexsize {
        let (path, nodes) = ht.softmax_path(id as u32);
        for i in 0..max_codelen {
            (vm.path[(id, i)], vm.code[(id, i)]) =
                if i < nodes.len() { (nodes[i], if path[i] { 1 } else { 0 }) }
                else { ( u32::MAX, u8::MAX ) };
        }
    }

    let outpath = args.outpath;

    let mut lexf = std::io::BufWriter::new(
        std::fs::File::create(outpath.to_string() + ".dict")?);
    for id in ixs.iter() {
        write!(lexf, "{} {}\n", attr.lex.id2str(*id), ofreqs[*id as usize])?;
    }
    lexf.flush()?;
    std::mem::drop(lexf);

    /*
    let mut frqf = std::io::BufWriter::new(
        std::fs::File::create(outpath.to_string() + ".frq")?);
    for frq in vm.freqs.iter() {
        frqf.write_all(&frq.to_le_bytes())?;
    }
    frqf.flush()?;
    std::mem::drop(frqf); */

    // let batch = 64000;
    let dim = args.dim;
    let prototypes = args.prototypes;

    let mut total_frq = 0u64;
    for frq in vm.freqs.iter() {
        total_frq += frq;
    }

    let init_count = -1;
    for w in 0..reduced_lexsize {
        vm.counts[[w, 0]] = if init_count > 0 { init_count as f32 } else { vm.freqs[w] as f32 };
    }

    let doc = corp.open_struct("doc", false)?;
    //let txt = attr.text;

    //let mut di = dociterm(&*doc, &*txt, &oid_to_nid);
    /* for _ in 0..10 {
        if let Some(v) = di.next() {
            println!("v {:?}", v);
            let pp = preprocess(&v, &vm.freqs.as_slice().unwrap(),
                                total_frq, 5, 1e-5, &mut rng);
            println!("pp {:?}", pp);
        } else {
            break;
        }
    } */

    let mut cntr = 0u64;
    let doc_cnt = doc.len();
    let mut doc_read = 0u64;
    let mut total_ll1 = 0.;
    let mut total_ll2 = 0.;
    let mut words_read = 0;

    let mut in_grad = Array::<f32, Ix2>::zeros((prototypes, dim));
    let mut out_grad = Array::<f32, Ix1>::zeros(dim);
    let mut z = Array::<f64, Ix1>::zeros(prototypes);

    let mut senses = 0;
    let mut max_senses = 0;
 
    let txt = attr.text;

    for rawdoc in dociterm(&*doc, &*txt, &oid_to_nid) {
        let doc = preprocess(&rawdoc, &vm.freqs.as_slice().unwrap(),
                              total_frq, args.min_freq,
                              args.subsample as f64, &mut rng);
        for i in 0..doc.len() {
            let x = doc[i];

            let lr1 = f64::max(
                args.start_lr * (1. - doc_read as f64 / (doc_cnt as f64+1.)),
                args.start_lr * 1e-4);
            let lr2 = lr1;

            // random reduce ... TODO
            let window = args.window as isize;

            let n_senses = var_init_z(&mut vm, x, &mut z);
            senses += n_senses;
            max_senses = std::cmp::max(max_senses, n_senses);
            for j in std::cmp::max(0, i as isize - window)..std::cmp::min(doc.len() as isize, i as isize + window) {
                if i as isize == j { continue; }
                let y = doc[j as usize];
                var_update_z(&mut vm, x, y, &mut z);
            }

            exp_normalize(&mut z);

            for j in std::cmp::max(0, i as isize - window)..std::cmp::min(doc.len() as isize, i as isize + window) {
                if i as isize == j { continue; }
                let y = doc[j as usize];
                let ll = in_place_update(&mut vm, x, y, &z, lr1, &mut in_grad,
                                         &mut out_grad, args.sense_threshold);
                total_ll2 += 1.;
                total_ll1 += (ll - total_ll1) / total_ll2;
            }

            words_read += 1;

            var_update_counts(&mut vm, x, &z, lr2);

            // timing TODO
            
            // if words_read > total_words break
        }

        //cntr += 1; if cntr > 5 { break; }
        doc_read += 1;
    }

    let lexx = attr.lex;
    let id2word = |id| lexx.id2str(ixs[id as usize]).to_string();

    save_model(&(outpath.to_string() + ".vm"), &vm, args.save_threshold, id2word)?;
    //println!("{:?}", ht.softmax_path(args.dim));
    //dbg!(ht.convert());
    Ok(())
}

fn expected_pi(vm: &VectorModel, w: u32, pi: &mut Array<f64, Ix1>) -> u32 {
    let min_prob = 1e-3f64;

    let mut r = 1.;
    let mut senses = 0;
    let mut ts = vm.counts.slice(s![w as usize, ..]).sum() as f64;
    
    for k in 0..vm.nmeanings()-1 {
        ts = f64::max(ts - vm.counts[[w as usize, k]] as f64, 0.);
        let a = 1. + vm.counts[[w as usize, k]] as f64;
        let b = vm.alpha + ts;
        pi[k] = mean_beta(a, b) * r;
        if pi[k] >= min_prob { senses += 1; }
        r = f64::max(r - pi[k], 0.)
    }
    pi[vm.nmeanings()-1] = r; 
    if r >= min_prob { senses += 1; }
    senses
}

fn var_init_z(vm: &mut VectorModel, w: u32, pi: &mut Array<f64, Ix1>) -> u32 {
    let min_prob = 1e-3f64;

    let mut r = 0f64;
    let mut x = 1f64;
    let mut senses = 0u32;
    for i in 0..pi.len() {
        pi[i] = vm.counts[[w as usize, i]] as f64;
    }
    let mut ts = pi.sum();
    for k in 0..pi.len()-1 {
        ts = f64::max(ts - pi[k], 0.);
        let a = 1. + pi[k];
        let b = vm.alpha + ts;
        pi[k] = meanlog_beta(a, b) + r;
        r += meanlog_mirror(a, b);

        let pi_k = mean_beta(a, b) * x;
        x = f64::max(x - pi_k, 0.);
        if pi_k >= min_prob {
            senses += 1;
        }
        if pi[k].is_nan() || pi[k] > 10e15 || pi[k] < -10e15 {
            panic!();
        }
    }
    let lp = pi.len();
    pi[lp-1] = r; 
    if x >= min_prob { senses += 1; }
    senses
}

fn sigmoid   (x: f64) -> f64 { 1. / (1.+(-x).exp()) }
fn logsigmoid(x: f64) -> f64 {     -(1.+(-x).exp()).ln() }

fn var_update_z(vm: &mut VectorModel, x: u32, y: u32, z: &mut Array<f64, Ix1>) {
    let t = vm.counts.len_of(Axis(1));
    let x = x as usize;
    let y = y as usize;
    
    for n in 0..vm.code.len_of(Axis(1)) {
        let code = vm.code[[y, n]];
        let path = vm.path[[y, n]] as usize;
        if code == u8::MAX { break; }

        let out_vec = vm.out_vecs.slice(s![path, ..]);

        for k in 0..t {
            let in_vec = vm.in_vecs.slice(s![x, k, ..]);
            let f = in_vec.dot(&out_vec) as f64;
            z[k] += logsigmoid(f * (1. - 2.*(code as f64)));
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
fn in_place_update(vm: &mut VectorModel, x: u32, y: u32, z: &Array<f64, Ix1>,
                   lr: f64,
                   in_grad: &mut Array<f32, Ix2>, out_grad: &mut Array<f32, Ix1>,
                   sense_threshold: f64) -> f64 {
    let mut pr = 0.;
    let t = vm.counts.len_of(Axis(1));
    let m = vm.in_vecs.len_of(Axis(2));
    let x = x as usize;
    let y = y as usize;

    in_grad.fill(0.);    

    for n in 0..vm.code.len_of(Axis(1)) {
        let code = vm.code[[y, n]];
        let path = vm.path[[y, n]] as usize;
        if code == u8::MAX { break; }

        //let mut out_vec = vm.out_vecs.slice_mut(s![path, ..]);
        let mut out_vec = vm.out_vecs.index_axis_mut(Axis(0), path);

        out_grad.fill(0.);

        for k in 0..t {
            if z[k] < sense_threshold { continue; }
            let in_vec = vm.in_vecs.slice(s![x, k, ..]);
            let f = out_vec.dot(&in_vec) as f64;
            
            pr += z[k] * logsigmoid(f * (1. - 2.*code as f64));
            let d = 1. - code as f64 - sigmoid(f);
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
        let mut in_vec = vm.in_vecs.slice_mut(s![x, k, ..]);
        //let mut in_vecc = vm.in_vecs.index_axis_mut(Axis(0), x);
        //let mut in_vec = in_vecc.index_axis_mut(Axis(0), k);
        in_vec += &in_grad.slice(s![k, ..]);
    }
    pr
}

fn var_update_counts(vm: &mut VectorModel, x: u32,
                     local_counts: &Array<f64, Ix1>, lr2: f64) {
    for k in 0..vm.counts.len_of(Axis(1)) {
        vm.counts[[x as usize, k]] += (lr2 *
            (local_counts[[k as usize]] * vm.freqs[[x as usize]] as f64 - vm.counts[[x as usize, k]] as f64)) as f32;
    }
}

fn preprocess(doc: &[u32], freqs: &[u64], total_frq: u64,
              min_freq: u64, subsampling_threshold: f64, rng: &mut SmallRng) -> Vec<u32> {
    let mut out = Vec::with_capacity(doc.len());
    let u = distributions::Uniform::<f64>::new(0., 1.);
    for id in doc {
        let f = freqs[*id as usize];
        if f < min_freq { continue; };
        if u.sample(rng)
            < 1. - (subsampling_threshold /
                    (f as f64 / total_frq as f64)).sqrt() {
            continue;
        }
        out.push(*id);
    }
    out
}

struct DocIter<'a> {
    docpos: usize,
    doc: &'a dyn corp::structure::Struct,
    text: &'a dyn corp::text::Text,
}

fn dociter<'a>(doc: &'a dyn corp::structure::Struct,
              text: &'a dyn corp::text::Text) -> DocIter<'a> {
    DocIter { docpos: 0, doc: doc, text: text }
}

struct DocIterM<'a> {
    di: DocIter<'a>,
    oid_to_nid: &'a[u32],
}

fn dociterm<'a>(doc: &'a dyn corp::structure::Struct,
              text: &'a dyn corp::text::Text,
              oid_to_nid: &'a[u32]) -> DocIterM<'a> {
    DocIterM { di: dociter(doc, text), oid_to_nid: oid_to_nid }
}

impl Iterator for DocIterM<'_> {
    type Item = Vec<u32>;
    fn next(&mut self) -> Option<Vec<u32>> {
        self.di.next().map(|v|
            v.iter().filter_map(|oid| {
                match self.oid_to_nid[*oid as usize] {
                    u32::MAX => None,
                    nid => Some(nid),
                }
            }).collect()
        )
    }
}

impl Iterator for DocIter<'_> {
    type Item = Vec<u32>;
    fn next(&mut self) -> Option<Vec<u32>> {
        if self.docpos < self.doc.len() {
            let beg = self.doc.beg_at(self.docpos as u64);
            let end = self.doc.end_at(self.docpos as u64);
            println!("{}: {}, {}", self.docpos, beg, end);
            let it = self.text.at(beg);
            let vals = it.take((end - beg) as usize).collect();
            self.docpos += 1;
            Some(vals)
        } else { None }
    }
}

fn exp_normalize(x: &mut Array<f64, Ix1>) {
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
}




