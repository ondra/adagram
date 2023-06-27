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

use std::collections::HashMap;

//#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use]
extern crate rocket;

const VERSION: &str = git_version::git_version!(args=["--tags","--always", "--dirty"]);

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

    /// HTTP listening port
    #[clap(long,default_value_t=7878)]
    port: u16,
}

struct ServerState {
    wslex: WSLex, 
    corpid2id: Vec<u32>,
    wmap: WMap,
    vm: VectorModel,
    sense_neighbors: usize,
    id2str: Vec<String>,
    window: usize,
    wsminrnk: f32,
    wsminfrq: u64,
    sampleconc: Option<u64>,
    attr: Box<dyn corp::corp::Attr>,
    defattr: Box<dyn corp::corp::Attr>,
}

use rayon::prelude::*;
#[get("/<head>")]
fn neighbors(head: String, state: &rocket::State<ServerState>) -> Result<rocket::response::content::RawJson<String>, String> {
    let head_cid = if let Some(head_cid) = state.wslex.head2id(&head) {
        head_cid
    } else {
        return Err(format!("ERROR: '{}' not found in WSATTR lexicon", head).into());
    };
    let head_mid = state.corpid2id[head_cid as usize];
    if head_mid == u32::MAX {
        return Err(format!("ERROR: unable to translate '{}' with WSATTR id {} to model id'", head, head_cid).into());
    }
    let nsenses = state.vm.in_vecs.len_of(Axis(1));

    // let mut out_senses: HashMap<usize, Vec<(String, u32, f32)>> = std::collections::HashMap::new();

    let out_senses: HashMap<usize, Vec<(String, u32, f32)>> =
            (0..nsenses).into_par_iter().map(|i| { 
        let r = nearest(&state.vm, head_mid as usize, i,
                        state.sense_neighbors, 5);
        print!("# sense {} ({}):", i, state.vm.counts[[head_mid as usize, i]]);
        let mut vec_neighbors = Vec::new();
        for (mid, senseno, sim) in r {
            vec_neighbors.push((state.id2str[mid as usize].to_string(), senseno, sim));
            print!("\t{}##{}/{:.3}", state.id2str[mid as usize], senseno, sim);
        }
        (i, vec_neighbors)
        // out_senses.insert(i, vec_neighbors);
    }).collect();

    match serde_json::to_string(&out_senses) {
        Ok(s) => Ok(rocket::response::content::RawJson(s)),
        Err(e) => Err(format!("failed to encode response as json: {}", e)),
    }
}

#[get("/<head>")]
fn desamb(head: String, state: &rocket::State<ServerState>) -> Result<rocket::response::content::RawJson<String>, String> {
    let ntokens = 2*state.window;
    // let mut rng = SmallRng::seed_from_u64(666);
        
    let head_cid = if let Some(head_cid) = state.wslex.head2id(&head) {
        head_cid
    } else {
        return Err(format!("ERROR: '{}' not found in WSATTR lexicon", head).into());
    };
    let head_mid = state.corpid2id[head_cid as usize];
    if head_mid == u32::MAX {
        return Err(format!("ERROR: unable to translate '{}' with WSATTR id {} to model id'", head, head_cid).into());
    }
    let headx = if let Some(headx) = state.wmap.find_id(head_cid) {
        headx
    } else {
        return Err(format!("ERROR: model and lexicon know '{}', but it is not present in word sketch", head).into());
    };

    let mut out_rel: HashMap<String, HashMap<String, (usize, f64, f64)>> = std::collections::HashMap::new();

    for relx in headx.iter() {
        let rels = state.wslex.id2rel(relx.id);

        // let mut out_coll = std::collections::HashMap::new();
        // let out_coll: std::collections::HashMap<String, (usize, f64, f64)> = relx.iter().filter_map(|collx| {
        let out_coll: std::collections::HashMap<String, (usize, f64, f64)> = relx.iter().par_bridge().filter_map(|collx| {
            let mut lctx = Vec::<u32>::with_capacity(ntokens);
            let mut rctx = Vec::<u32>::with_capacity(ntokens);
            let colls = state.wslex.id2coll(collx.id);
            if collx.cnt < state.wsminfrq || collx.rnk < state.wsminrnk {
                return None;
            }

            let mut z = Array::<f64, Ix1>::zeros(state.vm.nmeanings());
            let mut zst = vec![];
            for _ in 0..state.vm.nmeanings() {
                zst.push(RunningStats::new());
            }
            for zk in zst.iter_mut() { (*zk).clear(); }
            let it = collx.iter();
            let mut rng = SmallRng::seed_from_u64(
                ((collx.id as u64) << 10) + (relx.id as u64));
            let itf = || -> Box<dyn Iterator<Item=(usize, Option<i32>)>> {
                if let Some(nsamples) = state.sampleconc {
                    if nsamples < collx.cnt {
                        return Box::new(
                            it.sample(nsamples as usize, &mut rng)
                        )
                    }
                }
                Box::new(it)
            };

            for (pos, _coll) in itf() {
                let _n_senses = expected_pi(&state.vm.counts, state.vm.alpha,
                                            head_mid, &mut z);

                for zk in z.iter_mut() {
                    if *zk < 1e-3 { *zk = 0.; }
                    *zk = zk.ln();  // ???
                }

                let start = if pos >= ntokens { pos - ntokens } else { 0 };
                let ctxit = state.attr.iter_ids(start as u64);

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
                    match state.corpid2id[*corpid as usize] {
                        u32::MAX => None,
                        id => Some(id),
                    }
                };

                for ctx_mid in lctx.iter()
                    .rev().filter_map(fmap_ids).take(state.window) {
                    var_update_z(&state.vm.in_vecs, &state.vm.out_vecs, &state.vm.code,
                                 &state.vm.path, head_mid, ctx_mid, &mut z);
                }

                for ctx_mid in rctx.iter()
                    .filter_map(fmap_ids).take(state.window) {
                    var_update_z(&state.vm.in_vecs, &state.vm.out_vecs, &state.vm.code,
                                 &state.vm.path, head_mid, ctx_mid, &mut z);
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

            let oc = if let Some(mp) = maxpos {
                print!("\t{:.2}/{:.2}", zst[mp].mean(), zst[mp].stddev());
                (mp, zst[mp].mean(), zst[mp].stddev())
            } else {
                print!("\t{:.2}/{:.2}", -1., -1.);
                (999, -1., -1.)
            };
            return Some((colls.to_string(), oc));
            //out_coll.insert(colls.to_string(), oc);
            /*
            print!("\t{}", zst.iter()
                .enumerate()
                .map(|(i, p): (usize, &RunningStats)| -> String {
                    format!("{}:{:.2}/{:.2}", i, p.mean(), p.stddev())})
                .collect::<Vec<String>>()
                .join(" "));
            */

            //print!("{}\t{}\t{}\t{}\t{}\t",
            //       head, rels, colls, collx.cnt, collx.rnk);

            /*if collx.lcm.len() >= 2 {
                for i in 0..collx.lcm.len()-1 {
                    print!("{}", state.defattr.id2str(collx.lcm[i] as u32));
                    if i != collx.lcm.len()-2 {
                        print!(" ");
                    }
                }
            }
            println!();*/
        }).collect::<HashMap<String, (usize, f64, f64)>>();
        out_rel.insert(rels.to_string(), out_coll);
    }

    match serde_json::to_string(&out_rel) {
        Ok(s) => Ok(rocket::response::content::RawJson(s)),
        Err(e) => Err(format!("failed to encode response as json: {}", e)),
    }
}

#[rocket::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let mut config = rocket::Config::default();
    config.port = args.port;

    rocket::custom(config)
    .manage(ServerState{wslex, corpid2id, wmap, vm, sense_neighbors: args.sense_neighbors, id2str,
        window:args.window, wsminfrq:args.wsminfrq, wsminrnk:args.wsminrnk, sampleconc: args.sampleconc, attr, defattr})
    .mount("/neighbors/", routes![neighbors])
    .mount("/ws/", routes![desamb])
    .launch().await?;

    Ok(())
}

