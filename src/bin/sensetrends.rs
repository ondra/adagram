use clap::Parser;

use ndarray::prelude::*;

use adagram::diachronic::*;
use adagram::adagram::*;
use adagram::common::*;
use adagram::runningstats::RunningStats;

use rayon::prelude::*;

const VERSION: &str = git_version::git_version!(args=["--tags", "--always", "--dirty"]);

/// Assign Word Sketches to senses
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

    /// number of worker threads to use (0 to use all processors)
    #[clap(long, default_value_t=0)]
    nthreads: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let corpus = corp::corp::Corpus::open(&args.corpname)?;

    if args.nthreads != 0 {
        rayon::ThreadPoolBuilder::new().num_threads(args.nthreads).build_global().unwrap();
    }

    eprintln!("opening attribute {}", &args.posattr);
    let posattr = corpus.open_attribute(&args.posattr)?;
    eprintln!("opening attribute {}", &args.diaattr);
    let diaattrparts = &mut args.diaattr.split('.');
    let diastructname = diaattrparts.next().ok_or("")?;
    let diastructattr = corpus.open_attribute(&args.diaattr)?;
    let diastruct = corpus.open_struct(diastructname)?;

    let (diamap, new_norms) = map_diavals(diastructattr.as_ref(), args.epoch_limit)?;

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

    let mut zst = vec![];
    for _ in 0..vm.nmeanings() {
        zst.push(RunningStats::new());
    }

    let ntokens = 2*args.window;

    // diachronic init
    let h = posattr.id_range() as usize;
    let epochcnt = new_norms.len();
    eprintln!("w {}", epochcnt);

    if epochcnt < 2 {
        eprintln!("WARNING: fewer than 2 valid structattr values");
        return Err("semantic error".into());
    }

    if h <= 0 {
        eprintln!("WARNING: empty corpus");
        return Err("semantic error".into());
    }

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
                    use std::sync::{Arc, Mutex};
    let sense_diacnts = Arc::new(Mutex::new(vec![0f64; nmeanings * epochcnt]));

    eprintln!("ready");
    for line in std::io::stdin().lines() {
        {
            let mut sense_diacnts = sense_diacnts.lock().unwrap();
            sense_diacnts.iter_mut().for_each(|v| *v = 0.);
        }
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
                let z = Array::<f64, Ix1>::zeros(vm.nmeanings());
                let lctx = Vec::with_capacity(ntokens);
                let rctx = Vec::with_capacity(ntokens);
                (z, lctx, rctx)
            },
            |(z, lctx, rctx), pos|{
            let structpos = if let Some(structpos) = diastruct.num_at_pos(pos) {
                structpos
            } else {
                eprintln!("WARN: position {} for id {} is outside structure", pos, corpid);
                return None;
            };

            let n_senses = expected_pi(&vm.counts, vm.alpha, x, z, args.sense_threshold);

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

            let diaid = diastructattr.text().get(structpos);

            let epoch_no = diamap[diaid as usize];
            if epoch_no == u32::MAX {
                return None;
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
                    head_mid, ctx_mid, z);
            }

            for ctx_mid in rctx.iter()
                    .filter_map(fmap_ids).take(args.window) {
                var_update_z(&vm.in_vecs, &vm.out_vecs, &vm.code, &vm.path,
                    head_mid, ctx_mid, z);
            }

            exp_normalize(z);
            //for (i, zk) in z.iter().enumerate() {
            //    zst[i].push(*zk);

            let maxsense: Option<usize> = z.iter()
                .enumerate()
                .max_by(|(_, a),(_, b)| a.total_cmp(b))
                .map(|(i, _)| i);
            Some((maxsense.unwrap(), epoch_no))
        }).flatten();
        pit.for_each(|(maxsense, epoch_no)|{
            let mut sense_diacnts = sense_diacnts.lock().unwrap();
            sense_diacnts[maxsense*epochcnt + epoch_no as usize] += 1.;
        });
        let sense_diacnts = sense_diacnts.lock().unwrap();

        let freqs = (0..epochcnt)
            .map(|epoch|
                (0..nmeanings).map(|sense| sense_diacnts[sense*epochcnt + epoch]).sum()
            ).collect::<Vec<f64>>();

        println!("HW {}", head);
        for sense in 0..nmeanings as usize {
            print!("s##{}", sense);
            for epoch in 0..epochcnt {
                print!("\t{}", sense_diacnts[sense*epochcnt + epoch]);
            }
            println!();
        }
        print!("f");
        for epoch in 0..epochcnt {
            print!("\t{}", freqs[epoch]);
        }
        println!();
        print!("n");
        for epoch in 0..epochcnt {
            print!("\t{}", new_norms[epoch]);
        }
        println!();

        /*
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

    eprintln!("done.");
    Ok(())
}

