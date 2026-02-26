#[path = "../global_alloc.rs"]
mod global_alloc;

use clap::Parser;

use ndarray::Array;
use ndarray::prelude::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::distributions;
use ndarray_rand::rand::prelude::SmallRng;
use ndarray_rand::rand_distr::Distribution;

use adagram::adagram::VectorModel;
use adagram::common::*;
use adagram::hogwild;
use adagram::huffman;

const VERSION: &str = git_version::git_version!(args = ["--tags", "--always", "--dirty"]);

/// Train an adaptive skip-gram model
#[derive(Parser, Debug)]
#[clap(author, version=VERSION, about)]
struct Args {
    /// training corpus
    corpname: String,

    /// training attribute
    attrname: String,

    /// window size
    #[clap(long, default_value_t = 4)]
    window: u32,

    /// minimum frequency of the word
    #[clap(long, default_value_t = 20)]
    min_freq: u64,

    /// remove top K most frequent words
    #[clap(long, default_value_t = 0)]
    remove_top_k: u32,

    /// dimensionality of the representations
    #[clap(long, default_value_t = 100)]
    dim: usize,

    /// number of word prototypes
    #[clap(long, default_value_t = 5)]
    prototypes: usize,

    /// prior probability of allocating a new prototype
    #[clap(long, default_value_t = 0.1)]
    alpha: f64,

    /// subsampling threshold
    #[clap(long,default_value_t=f32::INFINITY)]
    subsample: f32,

    /// number of epochs to train
    #[clap(long, default_value_t = 1.0)]
    epochs: f64,

    /// randomly reduce size of the context
    #[clap(long)]
    context_cut: bool,

    /// initial weight (count) on first sense for each word; negative means use word frequency
    #[clap(long, default_value_t = -1.)]
    init_count: f32,

    /// minimal probability of a meaning to contribute into gradients
    #[clap(long, default_value_t = 1e-10)]
    sense_threshold: f64,

    /// minimal probability of a meaning to save after training
    #[clap(long, default_value_t = 1e-3)]
    save_threshold: f64,

    /// initial learning rate
    #[clap(long, default_value_t = 0.025)]
    start_lr: f64,

    /// path to save the output
    outpath: String,

    /// number of training threads to run in parallel
    #[clap(long, default_value_t = 1)]
    threads: usize,

    /// document structure -- do not cross begs or ends
    #[clap(long, default_value = "doc")]
    docstructure: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    adagram::common::init_math();

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
        if freq >= args.min_freq {
            ixs.push(id);
        }
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
        if a.len() > max_codelen {
            max_codelen = a.len();
        }
    }
    eprintln!("maximum code length is {}", max_codelen);

    let mut rng = SmallRng::seed_from_u64(666);
    let mut vm = VectorModel::new(
        args.dim,
        args.prototypes,
        reduced_lexsize,
        max_codelen,
        args.alpha,
        &mut rng,
    );

    vm.freqs.assign(&Array::from_vec(freqs));
    for id in 0..reduced_lexsize {
        let (path, nodes) = ht.softmax_path(id as u32);
        for i in 0..max_codelen {
            (vm.path[(id, i)], vm.code[(id, i)]) = if i < nodes.len() {
                (nodes[i], u8::from(path[i]))
            } else {
                (u32::MAX, u8::MAX)
            };
        }
    }

    let dim = vm.dim_padded;
    let prototypes = args.prototypes;

    let total_frq: u64 = vm.freqs.iter().sum();

    for w in 0..reduced_lexsize {
        vm.counts[[w, 0]] = if args.init_count > 0. {
            args.init_count
        } else {
            vm.freqs[w] as f32
        };
    }

    let doc = corp.open_struct(&args.docstructure)?;

    let doc_cnt = doc.len();

    let starttime = std::time::Instant::now();

    let positions_per_epoch: usize = vm.freqs.iter().map(|&f| f as usize).sum();
    eprintln!("will visit {} positions per epoch", positions_per_epoch);
    let total_words = (positions_per_epoch as f64 * args.epochs) as usize;
    eprintln!("{} positions in total", total_words);

    let words_read = std::sync::atomic::AtomicUsize::new(0);

    let in_vecs_m: hogwild::HogwildArray<f32, Ix3> = vm.in_vecs.into();
    let out_vecs_m: hogwild::HogwildArray<f32, Ix2> = vm.out_vecs.into();
    let counts_m: hogwild::HogwildArray<f32, Ix2> = vm.counts.into();

    let code = vm.code.view();
    let path = vm.path.view();
    let alpha = vm.alpha;

    let freqs = vm.freqs.view();

    let trainfunc = |mut in_vecs: hogwild::HogwildArray<f32, Ix3>,
                     mut out_vecs: hogwild::HogwildArray<f32, Ix2>,
                     mut counts: hogwild::HogwildArray<f32, Ix2>,
                     thread_id: usize| {
        let mut words_read_last = 0;
        let mut reporttime = std::time::Instant::now();
        let mut loc_rng = rng.clone();

        let mut in_grad = Array::<f32, Ix2>::zeros((prototypes, dim));
        let mut out_grad = Array::<f32, Ix1>::zeros(dim);
        assert_simd_aligned(dim, in_grad.as_ptr(), "in_grad");
        assert_simd_aligned(dim, out_grad.as_ptr(), "out_grad");
        let mut z = Array::<f64, Ix1>::zeros(prototypes);
        let mut doc_buf = Vec::new();

        let partsize = doc_cnt / args.threads;
        let startdoc = partsize * thread_id;
        let starttime = std::time::Instant::now();
        let mut total_ll1 = 0.0f64;
        let mut total_ll2 = 0.0f64;
        let compute_ll = thread_id == 0;
        let window = args.window as isize;

        const SYNC_INTERVAL: usize = 1024;
        let mut unsync_positions = 0usize;
        #[allow(unused_assignments)]
        let mut global_words_read = 0usize;
        let mut lr = args.start_lr;

        // Sync local progress to the global counter and return the global position count.
        let sync = |unsync: &mut usize| -> usize {
            let prev = words_read.fetch_add(*unsync, std::sync::atomic::Ordering::Relaxed);
            let global = prev + *unsync;
            *unsync = 0;
            global
        };

        for rawdoc in dociterm(doc.as_ref(), attr.as_ref(), &oid_to_nid, startdoc) {
            preprocess_into(
                &rawdoc,
                freqs.as_slice().unwrap(),
                total_frq,
                args.subsample as f64,
                &mut loc_rng,
                &mut doc_buf,
            );
            let doc = &doc_buf;
            let mut in_mut = in_vecs.as_mut().view_mut();
            let mut out_mut = out_vecs.as_mut().view_mut();
            let mut counts_mut = counts.as_mut().view_mut();

            for i in 0..doc.len() {
                // Periodically sync with the global counter
                if unsync_positions >= SYNC_INTERVAL {
                    global_words_read = sync(&mut unsync_positions);
                    lr = f64::max(
                        args.start_lr * (1. - global_words_read as f64 / (total_words as f64 + 1.)),
                        args.start_lr * 1e-4,
                    );

                    if compute_ll {
                        let dur = reporttime.elapsed().as_secs_f64();
                        if dur > 1.0 {
                            let rws = global_words_read - words_read_last;
                            words_read_last = global_words_read;
                            let wps = rws as f64 / dur;
                            let remaining_secs = if wps > 0.0 {
                                (total_words - global_words_read) as f64 / wps
                            } else {
                                0.0
                            };
                            reporttime = std::time::Instant::now();
                            let elapsed = reporttime.duration_since(starttime).as_secs();
                            eprint!(
                                "\r[{}] visited {} positions out of {} ({:.2} %), {:.0} wps, {:02}h:{:02}m remaining, lr {:.5} ll {:.7}",
                                elapsed,
                                global_words_read,
                                total_words,
                                global_words_read as f64 / total_words as f64 * 100.0,
                                wps,
                                remaining_secs as u64 / 3600,
                                (remaining_secs as u64 % 3600) / 60,
                                lr,
                                total_ll1,
                            );
                        }
                    }

                    if global_words_read >= total_words {
                        return;
                    }
                }

                let x = doc[i];
                let j_lo = (i as isize - window).max(0) as usize;
                let j_hi = (i as isize + window).min(doc.len() as isize) as usize;

                var_init_z(&counts_mut, alpha, x, &mut z);

                for j in j_lo..j_hi {
                    if j == i { continue; }
                    var_update_z(&in_mut, &out_mut, &code, &path, x, doc[j], &mut z);
                }

                exp_normalize(&mut z);

                for j in j_lo..j_hi {
                    if j == i { continue; }
                    let ll = in_place_update(
                        &mut in_mut, &mut out_mut, &counts_mut,
                        x, doc[j], &z, &code, &path, lr,
                        &mut in_grad, &mut out_grad,
                        args.sense_threshold, compute_ll,
                    );
                    if compute_ll {
                        total_ll2 += 1.;
                        total_ll1 += (ll - total_ll1) / total_ll2;
                    }
                }

                var_update_counts(&freqs, &mut counts_mut, x, &z, lr);
                unsync_positions += 1;
            }
        }
        // Flush any remaining unsync'd positions
        sync(&mut unsync_positions);
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
                let handle = std::thread::Builder::new()
                    .name(format!("worker{}", thread_id))
                    .spawn_scoped(scope, move || {
                        trainfunc(in_vecs_c, out_vecs_c, counts_c, thread_id_c)
                    })
                    .unwrap();
                handles.push(handle);
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
    let elapsed = starttime.elapsed().as_secs_f64();
    eprintln!(
        "FINISHED: read {} words in {} epochs, {:.0} wps",
        local_words_read,
        args.epochs,
        local_words_read as f64 / elapsed,
    );

    vm.in_vecs = std::sync::Arc::try_unwrap(in_vecs_m.into_inner())
        .expect("arc")
        .into_inner();
    vm.out_vecs = std::sync::Arc::try_unwrap(out_vecs_m.into_inner())
        .expect("arc")
        .into_inner();
    vm.counts = std::sync::Arc::try_unwrap(counts_m.into_inner())
        .expect("arc")
        .into_inner();
    let id2word = |id| attr.id2str(ixs[id as usize]).to_string();

    let extrainfo =
        VERSION.to_string() + " " + &std::env::args().collect::<Vec<String>>().join("\t");

    vm.save_model(&mut outfile, args.save_threshold, id2word, extrainfo)?;
    use std::io::Write;
    outfile.flush()?;
    std::mem::drop(outfile);
    std::fs::rename(tmpoutpath, args.outpath)?;

    Ok(())
}

fn preprocess_into(
    doc: &[u32],
    freqs: &[u64],
    total_frq: u64,
    subsampling_threshold: f64,
    rng: &mut SmallRng,
    out: &mut Vec<u32>,
) {
    out.clear();
    let u = distributions::Uniform::<f64>::new(0., 1.);
    for &id in doc {
        let f = freqs[id as usize];
        if u.sample(rng) < 1. - (subsampling_threshold / (f as f64 / total_frq as f64)).sqrt() {
            continue;
        }
        out.push(id);
    }
}

/// Iterates over documents, wrapping around to the beginning, mapping original IDs to new IDs.
struct DocIter<'a> {
    docpos: usize,
    doc: &'a dyn corp::structure::Struct,
    attr: &'a (dyn corp::corp::Attr + 'a),
    oid_to_nid: &'a [u32],
}

fn dociterm<'a>(
    doc: &'a dyn corp::structure::Struct,
    attr: &'a dyn corp::corp::Attr,
    oid_to_nid: &'a [u32],
    from: usize,
) -> DocIter<'a> {
    DocIter { docpos: from, doc, attr, oid_to_nid }
}

impl Iterator for DocIter<'_> {
    type Item = Vec<u32>;
    fn next(&mut self) -> Option<Vec<u32>> {
        if self.docpos >= self.doc.len() {
            self.docpos = 0;
        }
        let beg = self.doc.beg_at(self.docpos as u64);
        let end = self.doc.end_at(self.docpos as u64);
        let vals = self.attr.iter_ids(beg)
            .take((end - beg) as usize)
            .filter_map(|oid| match self.oid_to_nid[oid as usize] {
                u32::MAX => None,
                nid => Some(nid),
            })
            .collect();
        self.docpos += 1;
        Some(vals)
    }
}
