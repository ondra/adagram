use clap::Parser;

use ndarray::Array;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::SmallRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::distributions;
use ndarray_rand::rand_distr::Distribution;

use adagram::adagram::VectorModel;
use adagram::common::*;
use adagram::huffman;

/// Train an adaptive skip-gram model
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
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
    let attr: Box<dyn corp::corp::Attr> = corp.open_attribute(&args.attrname)?;

    let lexsize = attr.id_range();
    let mut ofreqs: Vec<u64> = vec![0u64; lexsize as usize];
    let mut ixs: Vec<u32> = Vec::new();
    for id in 0..lexsize {
        let freq = attr.frq(id);
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

    // let mut cntr = 0u64;
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
 
    let id2word = |id| attr.id2str(ixs[id as usize]).to_string();
    
    let starttime = std::time::Instant::now();

    let mut reporttime = std::time::Instant::now();
    let mut words_read_last = 0;

    for epoch in 0..args.epochs {
    for rawdoc in dociterm(doc.as_ref(), attr.as_ref(), &oid_to_nid) {
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

            let n_senses = var_init_z(&vm, x, &mut z);
            senses += n_senses;
            max_senses = std::cmp::max(max_senses, n_senses);
            for j in std::cmp::max(0, i as isize - window)..std::cmp::min(doc.len() as isize, i as isize + window) {
                if i as isize == j { continue; }
                let y = doc[j as usize];
                var_update_z(&vm, x, y, &mut z);
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
            
            let dur = reporttime.elapsed().as_secs();
            if dur > 60 {
                let rws = words_read - words_read_last;
                words_read_last = words_read;
                eprintln!("read {} words in epoch {}/{}, {} wps, {} spw",
                          words_read, epoch+1, args.epochs, rws as f32 / dur as f32,
                          senses as f32 / words_read as f32);
                reporttime = std::time::Instant::now();
            }
            // if words_read > total_words break
        }

        doc_read += 1;
    }

    vm.save_model(&(args.outpath.to_string() + ".part"), args.save_threshold, id2word)?;
    eprintln!("epoch {} of {} finished", epoch+1, args.epochs);         
    }

    eprintln!("FINISHED: read {} words in {} epochs, {} wps",
              words_read, args.epochs,
              words_read as f32 / starttime.elapsed().as_secs() as f32);

    vm.save_model(&(args.outpath.to_string()), args.save_threshold, id2word)?;
    //println!("{:?}", ht.softmax_path(args.dim));
    //dbg!(ht.convert());
    Ok(())
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

/*
struct DocIterM<'a> {
    docpos: usize,
    doc: &'a (dyn corp::structure::Struct + 'a),
    att: &'a (dyn corp::corp::Attr + 'a),
    oid_to_nid: &'a[u32],
}

fn dociterm<'a>(
              doc: &'a (dyn corp::structure::Struct + 'a),
              att: &'a (dyn corp::corp::Attr + 'a),
              oid_to_nid: &'a[u32],
              ) -> DocIterM<'a>
{
    DocIterM { docpos: 0,
        doc: doc, att: att,
        oid_to_nid: oid_to_nid }
}

impl Iterator for DocIterM<'_> {
    type Item = Vec<u32>;
    fn next(&mut self) -> Option<Vec<u32>> {
        if self.docpos < self.doc.len() {
            let beg = self.doc.beg_at(self.docpos as u64);
            let end = self.doc.end_at(self.docpos as u64);
            println!("{}: {}, {}", self.docpos, beg, end);
            let it = self.att.iter_ids(beg);
            //let it = vec![1u32,2,3].into_iter();
            let vals: Vec<u32> = it.take((end - beg) as usize).collect();
            self.docpos += 1;
            Some(vals.iter().filter_map(|oid| {
                match self.oid_to_nid[*oid as usize] {
                    u32::MAX => None,
                    nid => Some(nid),
                }
            }).collect())
        } else { None }
    }
}
*/


struct DocIter<'a> {
    docpos: usize,
    doc: &'a dyn corp::structure::Struct,
    attr: &'a (dyn corp::corp::Attr + 'a),
}

fn dociter<'a>(doc: &'a dyn corp::structure::Struct,
              attr: &'a (dyn corp::corp::Attr)) -> DocIter<'a>
{
    DocIter { docpos: 0, doc: doc, attr: attr }
}

struct DocIterM<'a> {
    di: DocIter<'a>,
    oid_to_nid: &'a[u32],
}

fn dociterm<'a>(doc: &'a dyn corp::structure::Struct,
              attr: &'a dyn corp::corp::Attr,
              oid_to_nid: &'a[u32]) -> DocIterM<'a>
{
    DocIterM { di: dociter(doc, attr), oid_to_nid: oid_to_nid }
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
            // println!("{}: {}, {}", self.docpos, beg, end);
            let it = self.attr.iter_ids(beg);
            let vals = it.take((end - beg) as usize).collect();
            self.docpos += 1;
            Some(vals)
        } else { None }
    }
}
