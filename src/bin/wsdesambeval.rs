use clap::Parser;

use ndarray::Array;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::SmallRng;
use ndarray_rand::rand::SeedableRng;

use adagram::adagram::VectorModel;
use adagram::common::*;
use adagram::nn::nearest;
use adagram::reservoir_sampling::SamplerExt;
use adagram::runningstats::RunningStats;

use corp::wsketch::WMap;
use corp::wsketch::WSLex;

const VERSION: &str = git_version::git_version!(args=["--tags", "--always", "--dirty"]);

/// Assign Word Sketches to senses 
#[derive(Parser, Debug)]
#[clap(author, version=VERSION, about)]
struct Args {
    /// word sketch corpus
    corpname: String,

    /// attribute to use, should be compatible with the attribute used to train the model
    attrname: String,

    /// adaptive skip-gram model
    model: String,

    /// window size, token count on both sides of KWIC used for desambiguation
    #[clap(long,default_value_t=4)]
    window: usize,

    /// minimal apriori sense probability
    #[clap(long,default_value_t=1e-3)]
    sense_threshold: f64,

    /// amount of nearest neighbors for sense vectors to retrieve 
    #[clap(long,default_value_t=6)]
    sense_neighbors: usize,

    /// drop collocates below this rank
    #[clap(long,default_value_t=-100.)]
    wsminrnk: f32,

    /// drop collocates below this rank
    #[clap(long,default_value_t=1)]
    wsminfrq: u64,

    /// ignore a maximal cluster for collocates below this frequency
    #[clap(long,default_value_t=5)]
    clusterminfrq: usize,

    /// ignore a maximal cluster for collocates below this probability
    #[clap(long,default_value_t=0.6)]
    clusterminprob: f64,

    /// visit at most the specified amount of concordance lines randomly
    #[clap(long)]
    sampleconc: Option<u64>,
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let corp = corp::corp::Corpus::open(&args.corpname)?;
    let attr: Box<dyn corp::corp::Attr> = corp.open_attribute(&args.attrname)?;

    let wsattrname = corp.get_conf("WSATTR").unwrap();
    let wsattr = corp.open_attribute(&wsattrname)?;
    let defattrname = corp.get_conf("DEFAULTATTR").unwrap();
    let defattr = corp.open_attribute(&defattrname)?;

    let wsbase = corp.get_conf("WSBASE").unwrap();
    let wmap = WMap::new(&wsbase)?;
    let wslex = WSLex::open(&wsbase, wsattr)?;

    eprintln!("loading model");
    let (vm, id2str) = VectorModel::load_model(&args.model)?;

    eprintln!("inverting model lexicon");
    let mut str2id = std::collections::HashMap::<&str, u32>
        ::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(word, id as u32);
    }

    eprintln!("mapping corpus and model lexicon");
    let mut corpid2id = Vec::<u32>::with_capacity(attr.id_range() as usize);
    for corpid in 0..attr.id_range() {
        let cval = attr.id2str(corpid);
        corpid2id.push(match str2id.get(cval) {
            Some(mid) => *mid,
            None => u32::MAX,
        });
    }
    assert!(corpid2id.len() == attr.id_range() as usize);

    let mut z = Array::<f64, Ix1>::zeros(vm.nmeanings());
    let mut zst = vec![];
    for _ in 0..vm.nmeanings() {
        zst.push(RunningStats::new());
    }

    let ntokens = 2*args.window;
    let mut lctx = Vec::<u32>::with_capacity(ntokens);
    let mut rctx = Vec::<u32>::with_capacity(ntokens);

    let mut rng = SmallRng::seed_from_u64(666);

    eprintln!("ready");
    for line in std::io::stdin().lines() {
        let unwrapped = line?;
        let head = unwrapped.trim();
        
        let head_cid = if let Some(head_cid) = wslex.head2id(head) {
            head_cid
        } else {
            eprintln!("ERROR: '{}' not found in WSATTR lexicon", head);
            println!("# ERROR: '{}' not found in WSATTR lexicon", head);
            continue;
        };

        let head_mid = corpid2id[head_cid as usize];
        if head_mid == u32::MAX {
            eprintln!("ERROR: unable to translate '{}' with WSATTR id {} to model id'", head, head_cid);
            println!("# ERROR: unable to translate '{}' with WSATTR id {} to model id'", head, head_cid);
            continue;
        }

        let headx = if let Some(headx) = wmap.find_id(head_cid) {
            headx
        } else {
            eprintln!("ERROR: model and lexicon know '{}', but it is not present in word sketch", head);
            println!("# ERROR: model and lexicon know '{}', but it is not present in word sketch", head);
            continue;
        };

        eprintln!("processing {}", head);

        let nsenses = vm.in_vecs.len_of(Axis(1));

        for i in 0..nsenses { 
            let r = nearest(&vm, head_mid as usize, i,
                            args.sense_neighbors, 5);
            print!("# sense {} ({}):", i, vm.counts[[head_mid as usize, i]]);
            for (mid, senseno, sim) in r {
                print!("\t{}##{}/{:.3}", id2str[mid as usize], senseno, sim);
            }
            println!();
        }
        
        for relx in headx.iter() {
            let rels = wslex.id2rel(relx.id);

            'coll: for collx in relx.iter() {
                let colls = wslex.id2coll(collx.id);
                if collx.cnt < args.wsminfrq || collx.rnk < args.wsminrnk {
                    continue 'coll;
                }

                for zk in zst.iter_mut() { (*zk).clear(); }
                let it = collx.iter();

                let itf = || -> Box<dyn Iterator<Item=(usize, Option<i32>)>> {
                    if let Some(nsamples) = args.sampleconc {
                        if nsamples < collx.cnt {
                            rng = SmallRng::seed_from_u64(
                                ((collx.id as u64) << 10) + (relx.id as u64));
                            return Box::new(
                                it.sample(nsamples as usize, &mut rng)
                            )
                        }
                    }
                    Box::new(it)
                };

                for (pos, _coll) in itf() {
                    let _n_senses = expected_pi(&vm.counts, vm.alpha,
                                                head_mid, &mut z, 1e-3f64);

                    for zk in z.iter_mut() {
                        if *zk < 1e-3 { *zk = 0.; }
                        *zk = zk.ln();  // ???
                    }

                    let start = if pos >= ntokens { pos - ntokens } else { 0 };
                    let ctxit = attr.iter_ids(start as u64);

                    lctx.clear(); rctx.clear();
                    for (ctxpos, ctx_cid) in std::iter::zip(start.., ctxit) {
                        if ctxpos == pos { continue; }
                        if ctxpos > pos + ntokens {
                            break;
                        }
                        if ctxpos < pos {
                            lctx.push(ctx_cid);
                        } else {
                            rctx.push(ctx_cid);
                        }
                    }

                    let fmap_ids = |corpid: &u32| {
                        match corpid2id[*corpid as usize] {
                            u32::MAX => None,
                            id => Some(id),
                        }
                    };

                    for ctx_mid in lctx.iter()
                        .rev().filter_map(fmap_ids).take(args.window) {
                        var_update_z(&vm.in_vecs, &vm.out_vecs, &vm.code,
                                     &vm.path, head_mid, ctx_mid, &mut z);
                    }

                    for ctx_mid in rctx.iter()
                        .filter_map(fmap_ids).take(args.window) {
                        var_update_z(&vm.in_vecs, &vm.out_vecs, &vm.code,
                                     &vm.path, head_mid, ctx_mid, &mut z);
                    }

                    exp_normalize(&mut z);

                    for (i, zk) in z.iter().enumerate() {
                        zst[i].push(*zk);
                    }

                    //print!("{}\t{}\t{}\t", wslex.id2head(headx.id), wslex.id2rel(relx.id), wslex.id2coll(collx.id));
                    //println!("{}", z.iter().map(|p: &f64|->String {format!("{:.4}", p)})
                    //    .collect::<Vec<String>>()  // won't be necessary in future rust
                    //    .join(" "));
                }

                let maxpos: Option<usize> = zst.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.mean().total_cmp(&b.mean()))
                    .map(|(i, _)| i);

                if let Some(mp) = maxpos {
                    print!("\t{:.2}/{:.2}", zst[mp].mean(), zst[mp].stddev());
                } else {
                    print!("\t{:.2}/{:.2}", -1., -1.);
                }

                /*
                print!("\t{}", zst.iter()
                    .enumerate()
                    .map(|(i, p): (usize, &RunningStats)| -> String {
                        format!("{}:{:.2}/{:.2}", i, p.mean(), p.stddev())})
                    .collect::<Vec<String>>()
                    .join(" "));
                */

                print!("{}\t{}\t{}\t{}\t{}\t",
                       head, rels, colls, collx.cnt, collx.rnk);

                if collx.lcm.len() >= 2 {
                    for i in 0..collx.lcm.len()-1 {
                        print!("{}", defattr.id2str(collx.lcm[i] as u32));
                        if i != collx.lcm.len()-2 {
                            print!(" ");
                        }
                    }
                }
                println!();
            }
        }
    }
    Ok(())
}

