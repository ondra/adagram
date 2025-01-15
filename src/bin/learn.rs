use clap::Parser;

use ndarray::Array;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::SmallRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::distributions;
use ndarray_rand::rand_distr::Distribution;

use adagram::adagram::VectorModel;
use adagram::common::*;
use adagram::hogwild;
use adagram::huffman;

const VERSION: &str = git_version::git_version!(args=["--tags","--always"]);

/// Train an adaptive skip-gram model
#[derive(Parser, Debug)]
#[clap(author, version=VERSION, about)]
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
    #[clap(long,default_value_t=1.0)]
    epochs: f64,

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

    /// initial learning rate
    #[clap(long,default_value_t=0.025)]
    start_lr: f64,

    /// path to save the output
    outpath: String,

    /// number of training threads to run in parallel
    #[clap(long,default_value_t=1)]
    threads: usize,

    /// document structure -- do not cross begs or ends
    #[clap(long,default_value="doc")]
    docstructure: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let tmpoutpath = args.outpath.to_string() + ".tmp";
    let mut outfile = std::fs::File::create(&tmpoutpath)?;

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
    eprintln!("pruned lexicon size is {}", reduced_lexsize);

    let mut freqs = vec![0u64; reduced_lexsize];
    let mut oid_to_nid = vec![u32::MAX; lexsize as usize];
    for (nid, oid) in ixs.iter().enumerate() {
        oid_to_nid[*oid as usize] = nid as u32;
        freqs[nid] = ofreqs[*oid as usize];
    }

    eprintln!("building huffman tree");
    let ht = huffman::HuffmanTree::new(&freqs);

    let mut max_codelen = 0;
    for id in 0..reduced_lexsize {
        let (a, _b) = ht.softmax_path(id as u32);
        if a.len() > max_codelen { max_codelen = a.len(); }
    }
    eprintln!("maximum code length is {}", max_codelen);

    let mut rng = SmallRng::seed_from_u64(666);
    let mut vm = VectorModel::new(args.dim, args.prototypes,
        reduced_lexsize, max_codelen, args.alpha, &mut rng);

    vm.freqs.assign(&Array::from_vec(freqs));
    for id in 0..reduced_lexsize {
        let (path, nodes) = ht.softmax_path(id as u32);
        for i in 0..max_codelen {
            (vm.path[(id, i)], vm.code[(id, i)]) =
                if i < nodes.len() { (nodes[i], u8::from(path[i])) }
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

    let doc = corp.open_struct(&args.docstructure)?;

    // let mut cntr = 0u64;
    let doc_cnt = doc.len();
    // let mut doc_read = 0u64;
    // let mut total_ll1 = 0.;
    // let mut total_ll2 = 0.;
    // let mut words_read = 0;

    // let mut senses = 0;
    // let mut max_senses = 0;
 
    let starttime = std::time::Instant::now();

    let mut total_words = 0usize;
    for (_id, cnt) in vm.freqs.iter().enumerate() {
        total_words += *cnt as usize;
    }
    eprintln!("will visit {} positions per epoch", total_words);
    total_words = (total_words as f64 * args.epochs) as usize;
    eprintln!("{} positions in total", total_words);
    let _total_words = total_words;

    let words_read = std::sync::atomic::AtomicUsize::new(0);

    let in_vecs_m: hogwild::HogwildArray<f32, Ix3> = vm.in_vecs.into();
    let out_vecs_m: hogwild::HogwildArray<f32, Ix2> = vm.out_vecs.into();
    let counts_m: hogwild::HogwildArray<f32, Ix2> = vm.counts.into();

    let code = vm.code.view();
    let path = vm.path.view();
    let alpha = vm.alpha;

    let freqs = vm.freqs.view();

    let trainfunc = |
            mut in_vecs: hogwild::HogwildArray<f32, Ix3>,
            mut out_vecs: hogwild::HogwildArray<f32, Ix2>,
            mut counts: hogwild::HogwildArray<f32, Ix2>,
            thread_id: usize,
        | {
        let mut words_read_last = 0;
        let mut reporttime = std::time::Instant::now();
        let mut loc_rng = rng.clone();

        let mut in_grad = Array::<f32, Ix2>::zeros((prototypes, dim));
        let mut out_grad = Array::<f32, Ix1>::zeros(dim);
        let mut z = Array::<f64, Ix1>::zeros(prototypes);

        let partsize = doc_cnt / args.threads;
        let startdoc = partsize * thread_id;
        let starttime = std::time::Instant::now();
        let mut total_ll1 = 0.0f64;
        let mut total_ll2 = 0.0f64;
        let mut local_words_read = 0;

        for rawdoc in dociterm(doc.as_ref(), attr.as_ref(), &oid_to_nid, startdoc) {
            let doc = preprocess(&rawdoc, freqs.as_slice().unwrap(),
                                  total_frq, args.min_freq,
                                  args.subsample as f64, &mut loc_rng);
            let mut in_mut = in_vecs.as_mut().view_mut();
            let mut out_mut = out_vecs.as_mut().view_mut();
            let mut counts_mut = counts.as_mut().view_mut();

            let lr1 = f64::max(
                args.start_lr * (1. - local_words_read as f64 / (total_words as f64+1.)),
                args.start_lr * 1e-4);
            let lr2 = lr1;

            if thread_id == 0 {
                let dur = reporttime.elapsed().as_secs_f64();
                if dur > 1.0 {
                    let rws = local_words_read - words_read_last;
                    words_read_last = local_words_read;
                    let wps = rws as f64 / dur;
                    let remaining_words = total_words - local_words_read;
                    let remaining_secs = if wps != 0.0 { remaining_words / wps as usize } else { 0 };
                    let remaining_hours = remaining_secs / 3600;
                    let remaining_mins = (remaining_secs % 3600) / 60;
                    reporttime = std::time::Instant::now();
                    let elapsed = reporttime.checked_duration_since(starttime).map(|d| d.as_secs()).unwrap_or(0);
                    eprint!("\r[{}] visited {} positions out of {} ({:.2} %), {:.0} wps, {:02}h:{:02}m remaining, lr {:.5} ll {:.7}", elapsed,
                              local_words_read, total_words, local_words_read as f64 / total_words as f64 * 100.0,
                              wps, remaining_hours, remaining_mins, lr1, total_ll1,
                              // senses as f32 / local_words_read as f32);
                              );
                }
            }

            for i in 0..doc.len() {
                let x = doc[i];

                // random reduce ... TODO
                let window = args.window as isize;

                let _n_senses = var_init_z(&counts_mut, alpha, x, &mut z);
                // senses += n_senses;
            
                // max_senses = std::cmp::max(max_senses, n_senses);
                for j in std::cmp::max(0, i as isize - window)..std::cmp::min(doc.len() as isize, i as isize + window) {
                    if i as isize == j { continue; }
                    let y = doc[j as usize];
                    var_update_z(&in_mut, &out_mut, &code, &path, x, y, &mut z);
                }

                exp_normalize(&mut z);

                for j in std::cmp::max(0, i as isize - window)..std::cmp::min(doc.len() as isize, i as isize + window) {
                    if i as isize == j { continue; }
                    let y = doc[j as usize];
                    let ll = in_place_update(&mut in_mut, &mut out_mut, &counts_mut,
                                             x, y, &z,
                                             &code, &path,
                                             lr1, &mut in_grad,
                                             &mut out_grad, args.sense_threshold);
                    total_ll2 += 1.;
                    total_ll1 += (ll - total_ll1) / total_ll2;
                }

                var_update_counts(&freqs, &mut counts_mut, x, &z, lr2);

            }
            local_words_read = words_read.fetch_add(doc.len(), std::sync::atomic::Ordering::Relaxed);
            if local_words_read >= total_words {
                return;
            }
        }
    };

    eprintln!("starting workers");

    if args.threads > 1 {
        std::thread::scope(|scope| {
            let mut handles = vec![];
            for thread_id in 0..args.threads {
                let in_vecs_c = in_vecs_m.clone();
                let out_vecs_c = out_vecs_m.clone();
                let counts_c = counts_m.clone();
                let thread_id_c = thread_id;
                handles.push(
                    scope.spawn(move ||
                        // trainfunc(in_vecs_m.clone(), out_vecs_m.clone(), counts_m.clone(), thread_id)
                        trainfunc(in_vecs_c, out_vecs_c, counts_c, thread_id_c)
                    )
                );
            }

            eprintln!("workers started");

            for handle in handles {
                match handle.join() {
                    Ok(()) => (),
                    Err(e) => std::panic::panic_any(e),
                }
            }
        })
    } else {
        trainfunc(in_vecs_m.clone(), out_vecs_m.clone(), counts_m.clone(), 0);
    }

    eprintln!();

    let local_words_read = words_read.load(std::sync::atomic::Ordering::Relaxed);
    // let local_words_read = local_words_read - args.threads;
    eprintln!("FINISHED: read {} words in {} epochs, {} wps",
              local_words_read, args.epochs,
              local_words_read as f32 / starttime.elapsed().as_secs() as f32);

    vm.in_vecs = std::sync::Arc::try_unwrap(in_vecs_m.into_inner()).expect("fuck").into_inner();
    vm.out_vecs = std::sync::Arc::try_unwrap(out_vecs_m.into_inner()).expect("fuck").into_inner();
    vm.counts = std::sync::Arc::try_unwrap(counts_m.into_inner()).expect("fuck").into_inner();
    let id2word = |id| attr.id2str(ixs[id as usize]).to_string();

    vm.save_model(&mut outfile, args.save_threshold, id2word)?;
    use std::io::Write;
    outfile.flush()?;
    std::mem::drop(outfile);
    std::fs::rename(tmpoutpath, args.outpath)?;

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


struct DocIter<'a> {
    docpos: usize,
    doc: &'a dyn corp::structure::Struct,
    attr: &'a (dyn corp::corp::Attr + 'a),
}

fn dociter<'a>(doc: &'a dyn corp::structure::Struct,
              attr: &'a (dyn corp::corp::Attr), from: usize) -> DocIter<'a>
{
    DocIter { docpos: from, doc, attr }
}

struct DocIterM<'a> {
    di: DocIter<'a>,
    oid_to_nid: &'a[u32],
}

fn dociterm<'a>(doc: &'a dyn corp::structure::Struct,
              attr: &'a dyn corp::corp::Attr,
              oid_to_nid: &'a[u32], from: usize) -> DocIterM<'a>
{
    DocIterM { di: dociter(doc, attr, from), oid_to_nid }
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

/* impl Iterator for DocIter<'_> {
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
} */

impl Iterator for DocIter<'_> {
    type Item = Vec<u32>;
    fn next(&mut self) -> Option<Vec<u32>> {
        if self.docpos >= self.doc.len() { self.docpos = 0; }
        let beg = self.doc.beg_at(self.docpos as u64);
        let end = self.doc.end_at(self.docpos as u64);
        // println!("{}: {}, {}", self.docpos, beg, end);
        let it = self.attr.iter_ids(beg);
        let vals = it.take((end - beg) as usize).collect();
        self.docpos += 1;
        Some(vals)
    }
}
