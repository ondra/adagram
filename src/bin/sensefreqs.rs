#[path = "../global_alloc.rs"]
mod global_alloc;

use clap::Parser;

use ndarray::prelude::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::prelude::SmallRng;

use adagram::adagram::*;
use adagram::common::*;
use adagram::diachronic::*;
use adagram::runningstats::RunningStats;

use rayon::prelude::*;

const VERSION: &str = git_version::git_version!(args = ["--tags", "--always", "--dirty"]);

/// Assign Word Form instances to senses and diachronic epochs
#[derive(Parser, Debug)]
#[clap(author, version=VERSION, about)]
struct Args {
    /// corpus
    corpname: String,

    /// positional attribute
    posattr: String,

    /// diachronic structure attribute
    diaattr: String,

    /// adaptive skip-gram model
    model: String,

    /// window size (if omitted, inferred from model path; fallback 10)
    #[clap(long)]
    window: Option<usize>,

    /// minimum apriori sense probability for the sense to be considered
    #[clap(long, default_value_t = 1e-3)]
    sense_threshold: f64,

    /// minimum norm for a structure attribute value to be considered
    #[clap(long, default_value_t = 0.15)]
    epoch_limit: f64,

    /// use uniform prior probabilities for senses
    #[clap(long, default_value_t = false)]
    uniform_prob: bool,

    /// accumulate sense distributions instead of counting the maximal values
    #[clap(long, default_value_t = false)]
    distrib: bool,

    /// number of worker threads to use (0 to use all processors)
    #[clap(long, default_value_t = 0)]
    nthreads: usize,

    /// do not print the TSV header as the first line of output
    #[clap(long, default_value_t = false)]
    skip_header: bool,

    /// skip positions with a specific attribute value (specify as attribute:value1,2,3,...)
    #[clap(long)]
    skip: Option<String>,

    /// visit at most the specified amount of concordance lines randomly
    #[clap(long)]
    sampleconc: Option<usize>,

    /// subcorpus file path (binary file of sorted (u64,u64) range pairs)
    #[clap(long)]
    subcorpus: Option<String>,

    /// print informative progress messages to stderr
    #[clap(short, long, default_value_t = false)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let corpus: Box<dyn corp::corp::CorpusLike> = match &args.subcorpus {
        Some(subcpath) => Box::new(corp::subcorp::SubCorpus::from_corpus(
            corp::corp::Corpus::open(&args.corpname)?, subcpath)?),
        None => Box::new(corp::corp::Corpus::open(&args.corpname)?),
    };

    let tpb = rayon::ThreadPoolBuilder::new().thread_name(|tid| format!("rayon_worker{}", tid));
    if args.nthreads != 0 {
        tpb.num_threads(args.nthreads)
    } else {
        tpb
    }
    .build_global()
    .unwrap();

    if args.verbose {
        eprintln!("opening attribute {}", &args.posattr);
    }
    let posattr = corpus.open_attribute(&args.posattr)?;
    let sampler = posattr.id2poss_sampler(args.sampleconc);
    if args.verbose {
        eprintln!("opening attribute {}", &args.diaattr);
    }
    let diaattrparts = &mut args.diaattr.split('.');
    let diastructname = diaattrparts.next().ok_or("")?;
    let diastructattr = corpus.open_attribute(&args.diaattr)?;
    let diastruct = corpus.open_struct(diastructname)?;
    let skip = if let Some(skip) = &args.skip {
        let skipparts = skip.splitn(2, ":").collect::<Vec<_>>();
        if skipparts.len() != 2 {
            return Err("bad skip specification".into());
        }
        let skipattrname = skipparts[0];
        let skipattr = corpus.open_attribute(skipattrname)?;

        let skipvals = skipparts[1];
        let mut skipids = std::collections::HashSet::<u32>::new();
        for skipval in skipvals.split(",") {
            if let Some(skipid) = skipattr.str2id(skipval) {
                skipids.insert(skipid);
            } else {
                eprintln!("WARNING: skip value {} not found in lexicon", skipval);
            }
        }
        Some((skipattr, skipids))
    } else {
        None
    };

    let (diamap, new_norms, ordered_epochnames) =
        map_diavals(diastructattr.as_ref(), args.epoch_limit)?;

    if args.verbose {
        eprintln!("loading model");
    }
    let (vm, id2str) = VectorModel::load_model(&args.model)?;

    if args.verbose {
        eprintln!("inverting model lexicon");
    }
    let mut str2id = std::collections::HashMap::<&str, u32>::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(word, id as u32);
    }

    if args.verbose {
        eprintln!("mapping corpus and model lexicon");
    }
    let mut corpid2id = Vec::<u32>::with_capacity(posattr.id_range() as usize);
    for corpid in 0..posattr.id_range() {
        let cval = posattr.id2str(corpid);
        corpid2id.push(match str2id.get(cval) {
            Some(mid) => *mid,
            None => u32::MAX,
        });
    }
    assert!(corpid2id.len() == posattr.id_range() as usize);

    let fmap_ids = |corpid: &u32| match corpid2id[*corpid as usize] {
        u32::MAX => None,
        id => Some(id),
    };

    let mut rng = SmallRng::seed_from_u64(666);

    let mut zst = vec![];
    for _ in 0..vm.nmeanings() {
        zst.push(RunningStats::new());
    }

    let window = parse_window(args.window, &args.model).unwrap_or(10);
    let ntokens = 2 * window;

    // diachronic init
    let h = posattr.id_range() as usize;
    let epochcnt = new_norms.len();
    if args.verbose {
        eprintln!("There are {} salient epochs to process.", epochcnt);
    }

    if epochcnt < 2 {
        eprintln!("WARNING: fewer than 2 valid structattr values");
        return Err("semantic error".into());
    }

    if h <= 0 {
        eprintln!("WARNING: empty corpus");
        return Err("semantic error".into());
    }

    let nmeanings = vm.nmeanings() as usize;

    if !args.skip_header {
        print!("hw\tepoch");
        for sense in 0..nmeanings {
            print!("\ts{}", sense);
        }
        println!("\tnorm");
    }

    if args.verbose {
        eprintln!("ready");
    }
    for line in std::io::stdin().lines() {
        let unwrapped = line?;
        let head = unwrapped.trim();

        let x = match str2id.get(head) {
            Some(n) => *n,
            None => {
                eprintln!("ERROR: {} not found in model lexicon", head);
                continue;
            }
        };

        let head_mid = x;

        let corpid = if let Some(cid) = posattr.str2id(head) {
            cid
        } else {
            eprintln!("ERROR: {} not found in corpus lexicon", head);
            continue;
        };

        let poss = sampler.id2poss_with_rng(corpid, &mut rng);
        let sense_diacnts = poss
            .par_bridge()
            .fold(
                || {
                    let lctx = Vec::with_capacity(ntokens);
                    let rctx = Vec::with_capacity(ntokens);
                    let z = Array::<f64, Ix1>::zeros(vm.nmeanings());
                    let sense_diacnts = vec![0f64; nmeanings * epochcnt];
                    (lctx, rctx, z, sense_diacnts)
                },
                |(mut lctx, mut rctx, mut z, mut sense_diacnts), pos| {
                    z.fill(0.0);
                    let structpos = if let Some(structpos) = diastruct.num_at_pos(pos) {
                        structpos
                    } else {
                        eprintln!(
                            "WARN: position {} for id {} is outside structure",
                            pos, corpid
                        );
                        return (lctx, rctx, z, sense_diacnts);
                    };

                    if let Some((skipattr, skipids)) = &skip {
                        let curskipattrid = skipattr.text().get(structpos);
                        if skipids.contains(&curskipattrid) {
                            return (lctx, rctx, z, sense_diacnts);
                        }
                    }

                    let n_senses =
                        expected_pi(&vm.counts, vm.alpha, x, &mut z, args.sense_threshold);

                    if args.uniform_prob {
                        for zk in z.iter_mut() {
                            if *zk < args.sense_threshold {
                                *zk = 0.;
                            } else {
                                *zk = 1. / n_senses as f64;
                            }
                        }
                    } else {
                        for zk in z.iter_mut() {
                            if *zk < args.sense_threshold {
                                *zk = 0.;
                            }
                            *zk = zk.ln();
                        }
                    }

                    let diaid = diastructattr.text().get(structpos);
                    let epoch_no = diamap[diaid as usize];
                    if epoch_no == u32::MAX {
                        return (lctx, rctx, z, sense_diacnts);
                    }

                    //eprintln!("{}", pos);
                    //if pos & 0xfff == 0 && Instant::now() >= next_report_time {
                    //    eprintln!("visited {} positions out of {} ({:.2} %)",
                    //        pos, h, 100.*(pos as f64)/(text_size as f64));
                    //    next_report_time += Duration::from_secs(240);
                    //}
                    let start = if pos >= ntokens as u64 {
                        pos - ntokens as u64
                    } else {
                        0
                    };
                    let ctxit = posattr.iter_ids(start);

                    lctx.clear();
                    rctx.clear();
                    for (ctxpos, ctx_cid) in std::iter::zip(start.., ctxit) {
                        if ctxpos == pos {
                            continue;
                        }
                        if ctxpos > pos + ntokens as u64 {
                            break;
                        }
                        if ctxpos < pos {
                            lctx.push(ctx_cid);
                        } else {
                            rctx.push(ctx_cid);
                        }
                    }

                    for ctx_mid in lctx.iter().rev().filter_map(fmap_ids).take(window) {
                        var_update_z(
                            &vm.in_vecs,
                            &vm.out_vecs,
                            &vm.code,
                            &vm.path,
                            head_mid,
                            ctx_mid,
                            &mut z,
                        );
                    }

                    for ctx_mid in rctx.iter().filter_map(fmap_ids).take(window) {
                        var_update_z(
                            &vm.in_vecs,
                            &vm.out_vecs,
                            &vm.code,
                            &vm.path,
                            head_mid,
                            ctx_mid,
                            &mut z,
                        );
                    }

                    exp_normalize(&mut z);

                    if !args.distrib {
                        let maxsense: usize = z
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .map(|(i, _)| i)
                            .unwrap();
                        sense_diacnts[maxsense * epochcnt + epoch_no as usize] += 1.0;
                    } else {
                        for iz in 0..z.len() {
                            sense_diacnts[iz * epochcnt + epoch_no as usize] += z[iz];
                        }
                    }

                    (lctx, rctx, z, sense_diacnts)
                },
            )
            .map(|(_lctx, _rctx, _z, sense_diacnts)| sense_diacnts)
            .reduce(
                || vec![0f64; nmeanings * epochcnt],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b) {
                        *ai += bi;
                    }
                    a
                },
            );

        for epoch in 0..epochcnt {
            print!("{}\t{}", head, ordered_epochnames[epoch]);
            for sense in 0..nmeanings {
                print!("\t{}", sense_diacnts[sense * epochcnt + epoch]);
            }
            println!("\t{}", new_norms[epoch]);
        }

        /*
        let freqs = (0..epochcnt)
            .map(|epoch|
                (0..nmeanings).map(|sense| sense_diacnts[sense*epochcnt + epoch]).sum()
            ).collect::<Vec<f64>>();
        let mut sum_normed = 0.0f64;
        for j in 0..w {
            normed[j] = if new_norms[j] != 0.0f64 {
                    &freqs[attr_id*w + j] / new_norms[j]
                } else { 0.0f64 };
            sum_normed += normed[j];
        }
        if sum_normed == 0.0f64 {
            eprintln!("zero sum_normed for id {}", attr_id);
            continue;
        }
        for j in 0..w {
            rel[j] = w as f64 * normed[j] / sum_normed;
        }

        let (lp, lslope) = adagram::diachronic::linreg(&xs[..], &rel);
        let (mp, mslope) = adagram::diachronic::mk(&xs[..], &rel);
        */
    }

    if args.verbose {
        eprintln!("done.");
    }
    Ok(())
}
