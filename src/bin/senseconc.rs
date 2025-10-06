use clap::Parser;

use ndarray::prelude::*;

use adagram::adagram::*;
use adagram::common::*;

use rayon::prelude::*;

const VERSION: &str = git_version::git_version!(args=["--tags", "--always", "--dirty"]);

/// Assign to senses to concordance lines
#[derive(Parser, Debug)]
#[clap(author, version=VERSION, about)]
struct Args {
    /// corpus
    corpname: String,
    
    /// positional attribute
    posattr: String,
    
    /// diachronic structure attribute
    diaattr: String,
    
    #[clap(long,default_value_t=5)]
    minfreq: usize,

    /// adaptive skip-gram model
    model: String,

    /// window size
    #[clap(long,default_value_t=4)]
    window: usize,

    /// minimum apriori sense probability for the sense to be considered
    #[clap(long,default_value_t=1e-3)]
    sense_threshold: f64,

    /// minimum norm for a structure attribute value to be considered
    #[clap(long,default_value_t=0.15)]
    epoch_limit: f64,

    /// use uniform prior probabilities for senses
    #[clap(long, default_value_t=false)]
    uniform_prob: bool,

    /// accumulate sense distributions instead of counting the maximal values
    #[clap(long, default_value_t=false)]
    distrib: bool,

    /// number of worker threads to use (0 to use all processors)
    #[clap(long, default_value_t=0)]
    nthreads: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let corpus = corp::corp::Corpus::open(&args.corpname)?;

    let tpb = rayon::ThreadPoolBuilder::new()
        .thread_name(|tid| format!("rayon_worker{}", tid));
    if args.nthreads != 0 { tpb.num_threads(args.nthreads) } else { tpb }
        .build_global().unwrap();

    eprintln!("opening attribute {}", &args.posattr);
    let posattr = corpus.open_attribute(&args.posattr)?;

    eprintln!("loading model");
    let (vm, id2str) = VectorModel::load_model(&args.model)?;
    
    eprintln!("inverting model lexicon");
    let mut str2id = std::collections::HashMap::<&str, u32>
        ::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(word, id as u32);
    }

    eprintln!("mapping corpus and model lexicon");
    let mut corpid2id = Vec::<u32>::with_capacity(posattr.id_range() as usize);
    for corpid in 0..posattr.id_range() {
        let cval = posattr.id2str(corpid);
        corpid2id.push(match str2id.get(cval) {
            Some(mid) => *mid,
            None => u32::MAX,
        });
    }
    assert!(corpid2id.len() == posattr.id_range() as usize);

    let fmap_ids = |corpid: &u32| {
        match corpid2id[*corpid as usize] {
            u32::MAX => None,
            id => Some(id),
        }
    };

    let ntokens = 2*args.window;

    // diachronic init
    //let mut normed = vec![0.0f64; epochcnt];
    //let mut rel = vec![0.0f64; epochcnt];
    //let mut next_report_time = Instant::now() + Duration::from_secs(60);
    //let xs = (0..epochcnt).map(|x| x as f64).collect::<Vec<_>>();

    // let mut next_report_time = Instant::now() + Duration::from_secs(60);
    /*
    let mut mktf = std::fs::File::create(trendbase.clone() + ".mkts_all.trends")?;
    let mut mktwr = BinaryTrendsWriter::new(&mut mktf)?;
    let mut mkmf = std::fs::File::create(trendbase.clone() + ".mkts_all.minigraphs")?;
    let mut mkmwr = BinaryMinigraphWriter::new(&mut mkmf)?;

    let mut lrtf = std::fs::File::create(trendbase.clone() + ".linreg_all.trends")?;
    let mut lrtwr = BinaryTrendsWriter::new(&mut lrtf)?;
    let mut lrmf = std::fs::File::create(trendbase.clone() + ".linreg_all.minigraphs")?;
    let mut lrmwr = BinaryMinigraphWriter::new(&mut lrmf)?;
    */

    let nmeanings = vm.nmeanings() as usize;

    eprintln!("ready");
    
    for line in std::io::stdin().lines() {
        let unwrapped = line?;
        let head = unwrapped.trim();

        let x = match str2id.get(head) {
            Some(n) => {
                *n
            },
            None => {
                eprintln!("ERROR: {} not found in model lexicon", head);
                continue;
            },
        };

        let head_mid = x;

        let corpid = if let Some(cid) = posattr.str2id(head) {
            cid
        } else {
            eprintln!("ERROR: {} not found in corpus lexicon", head);
            continue;
        };

        let poss = posattr.revidx().id2poss(corpid);
        let pit = poss.par_bridge().map_init(
            || {
                let lctx = Vec::with_capacity(ntokens);
                let rctx = Vec::with_capacity(ntokens);
                (lctx, rctx)
            },
            |(lctx, rctx), pos|{
            let mut z = Array::<f64, Ix1>::zeros(vm.nmeanings());
            let n_senses = expected_pi(&vm.counts, vm.alpha, x, &mut z, args.sense_threshold);

            if args.uniform_prob {
                for zk in z.iter_mut() {
                    if *zk < args.sense_threshold { *zk = 0.; }
                    else { *zk = 1. / n_senses as f64; }
                }
            } else {
                for zk in z.iter_mut() {
                    if *zk < args.sense_threshold { *zk = 0.; }
                    *zk = zk.ln();
                }
            }

            //eprintln!("{}", pos);
            //if pos & 0xfff == 0 && Instant::now() >= next_report_time {
            //    eprintln!("visited {} positions out of {} ({:.2} %)",
            //        pos, h, 100.*(pos as f64)/(text_size as f64));
            //    next_report_time += Duration::from_secs(240);
            //}
            let start = if pos >= ntokens as u64 { pos - ntokens as u64 } else { 0 };
            let ctxit = posattr.iter_ids(start);

            lctx.clear(); rctx.clear();
            for (ctxpos, ctx_cid) in std::iter::zip(start.., ctxit) {
                if ctxpos == pos { continue; }
                if ctxpos > pos + ntokens as u64 {
                    break;
                }
                if ctxpos < pos {
                    lctx.push(ctx_cid);
                } else {
                    rctx.push(ctx_cid);
                }
            }

            for ctx_mid in lctx.iter()
                    .rev().filter_map(fmap_ids).take(args.window) {
                var_update_z(&vm.in_vecs, &vm.out_vecs, &vm.code, &vm.path,
                    head_mid, ctx_mid, &mut z);
            }

            for ctx_mid in rctx.iter()
                    .filter_map(fmap_ids).take(args.window) {
                var_update_z(&vm.in_vecs, &vm.out_vecs, &vm.code, &vm.path,
                    head_mid, ctx_mid, &mut z);
            }

            exp_normalize(&mut z);
            Some((z, pos))
        }).flatten();

        /*pit.for_each(|(z, pos)|{
            for iz in 0..z.len() {
                sense_diacnts[pos] += z[iz];
            }
        });*/

        let max_rows = 100;
        let mut res: Vec<_> = pit
            .map(|(z, pos)| {
                let maxpos: usize = z.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i).unwrap();
                (z[maxpos], maxpos, pos)
            })
            .collect();

        res.sort_by(|(prob1, _maxpos1, _pos1), (prob2, _maxpos2, _pos2)| f64::total_cmp(prob1, prob2));

        for iz in 0..nmeanings {
            let mut found_rows = 0;
            for (prob, maxpos, pos) in &res {
                if *maxpos == iz {
                    found_rows += 1;
                    print!("{} {} {}", *prob, *maxpos, *pos);
                    if max_rows >= found_rows { break };
                }
            }
            // print!("\t{}", z[iz]);
        }

        println!("HW {}", head);
    }

    eprintln!("done.");
    Ok(())
}

