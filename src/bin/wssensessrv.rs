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
    /// tab-separated file containing pairs mapping language to model
    modelfile: String,

    /// window size, token count on both sides of KWIC used for desambiguation
    #[clap(long,default_value_t=4)]
    window: usize,

    /// minimal apriori sense probability
    #[clap(long,default_value_t=1e-3)]
    sense_threshold: f64,

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
    #[clap(long,default_value_t=0)]
    sampleconc: u64,

    /// HTTP listening port
    #[clap(long,default_value_t=9494)]
    port: u16,
}

type ModelMap = HashMap<String, (VectorModel, Vec<String>, HashMap<String, u32>)>;

struct ServerState {
    window: usize,
    wsminrnk: f32,
    wsminfrq: u64,
    sampleconc: u64,
    cmaps: std::sync::RwLock<HashMap<String, Vec<u32>>>,
    models: std::sync::Arc<std::sync::RwLock<ModelMap>>,
}

fn load_model(modellang: &str, modelpath: &str,
              models: &std::sync::RwLock<ModelMap>,
        ) -> Result<(), Box<dyn std::error::Error>> {
    let has_model = {
        let mr = models.read().unwrap();
        mr.get(modellang).is_some()
    };
    if !has_model {
        eprintln!("loading model");
        let (vm, id2str) = VectorModel::load_model(modelpath)?;

        eprintln!("inverting model lexicon");
        let str2id = id2str.iter().enumerate().par_bridge()
            .map(|(id, word)| { (word.to_string(), id as u32) })
            .collect::<HashMap<String, u32>>();

        let mut mw = models.write().unwrap();
        mw.insert(modellang.to_string(), (vm, id2str, str2id));
    }
    Ok(())
}

fn unload_model(modellang: &str,
                models: &std::sync::RwLock<ModelMap>,
        ) -> Result<(), Box<dyn std::error::Error>> {
    let mut mw = models.write().unwrap();
    match mw.remove(modellang) {
        Some(_model) => Ok(()),
        None => Err(format!("model {} not loaded", modellang).into()),
    }
}

#[delete("/<language>")]
fn req_unload_model(language: String, state: &rocket::State<ServerState>)
        -> Result<(), String> {
    unload_model(&language, &state.models).map_err(|e| { format!("{}", e) })
}

#[post("/<language>?<path>")]
fn req_load_model(language: String, path: Option<String>, state: &rocket::State<ServerState>)
        -> Result<(), String> {
    load_model(&language, &path.ok_or("parameter 'path' not present")?, &state.models)
        .map_err(|e| { format!("{}", e) })
}

#[get("/")]
fn req_list_models(state: &rocket::State<ServerState>)
        -> Result<rocket::response::content::RawJson<String>, String> {
    let mr = state.models.read().unwrap();
    match serde_json::to_string(&mr.keys().collect::<Vec<_>>()) {
        Ok(s) => Ok(rocket::response::content::RawJson(s)),
        Err(e) => Err(format!("failed to encode response as json: {}", e)),
    }
}

use rayon::prelude::*;
#[get("/<head>?<neighbors>&<language>")]
fn req_neighbors(head: String, neighbors: Option<usize>, language: String, state: &rocket::State<ServerState>)
        -> Result<rocket::response::content::RawJson<String>, String> {
    let mr = state.models.read().unwrap();
    let (vm, id2str, str2id) = match mr.get(&language) {
        Some(x) => x,
        None => {
            return Err(format!("model for {} not loaded", &language));
        },
    };
    let head_mid = match str2id.get(&head) {
        Some(mid) => *mid,
        None => {
            return Err(format!("could not translate {}", &head));
        },
    };

    let nsenses = vm.in_vecs.len_of(Axis(1));

    let out_senses: HashMap<usize, (Vec<(String, u32, f32, f32)>, f32)> =
            (0..nsenses).into_par_iter().map(|i| { 
        let min_count = 5;
        let r = nearest(vm, head_mid as usize, i,
                        neighbors.unwrap_or(5), min_count);
        let modelcount = vm.counts[[head_mid as usize, i]];
        // print!("# sense {} ({}):", i, state.vm.counts[[head_mid as usize, i]]);
        let mut vec_neighbors = Vec::new();
        for (mid, senseno, sim) in r {
            vec_neighbors.push((id2str[mid as usize].to_string(), senseno, sim, vm.counts[[mid as usize, senseno as usize]]));
            // print!("\t{}##{}/{:.3}", state.id2str[mid as usize], senseno, sim);
        }
        (i, (vec_neighbors, modelcount))
    }).collect();

    match serde_json::to_string(&out_senses) {
        Ok(s) => Ok(rocket::response::content::RawJson(s)),
        Err(e) => Err(format!("failed to encode response as json: {}", e)),
    }
}

#[allow(clippy::too_many_arguments)]
#[get("/<head>?<corpname>&<language>&<window>&<wsminrnk>&<wsminfrq>&<sampleconc>")]
fn req_wsdesamb(head: String, language: String, corpname: String, window: Option<usize>,
            wsminrnk: Option<f32>, wsminfrq: Option<u64>, sampleconc: Option<u64>,
            state: &rocket::State<ServerState>) -> Result<rocket::response::content::RawJson<String>, String> {
    let wsminrnk = wsminrnk.unwrap_or(state.wsminrnk);
    let wsminfrq = wsminfrq.unwrap_or(state.wsminfrq);
    let window = window.unwrap_or(state.window);
    let sampleconc = sampleconc.unwrap_or(state.sampleconc);
    let ntokens = 2*window;
    let mr = state.models.read().unwrap();
    let (vm, _id2str, str2id) = match mr.get(&language) {
        Some(x) => x,
        None => {
            return Err(format!("model for {} not loaded", &language));
        },
    };

    let corp = corp::corp::Corpus::open(&corpname).map_err(|e| format!("unable to open corpus: {}", e))?;
    let wsattrname = corp.get_conf("WSATTR").unwrap();
    let wsattr = corp.open_attribute(&wsattrname).map_err(|e| format!("unable to open WSATTR: {}", e))?;
    let wsbase = corp.get_conf("WSBASE").unwrap();
    let wmap = WMap::new(&wsbase).map_err(|e| format!("unable to open WMap: {}", e))?;

    {
        let has_c2m = {
            state.cmaps.read().unwrap().get(&corpname).is_some()
        };
        if !has_c2m {
            let mut mw = state.cmaps.write().unwrap();

            eprintln!("mapping corpus and model lexicon");
            let corpid2id = (0..wsattr.id_range()).into_par_iter().map(|corpid| {
                let cval = wsattr.id2str(corpid);
                *str2id.get(cval).unwrap_or(&u32::MAX)
            }).collect::<Vec<u32>>();

            mw.insert(corpname.clone(), corpid2id);
        }
    };

    let wslex = WSLex::open(&wsbase, wsattr).map_err(|e| format!("unable to open WS lexicon: {}", e))?;
    let wsattr = corp.open_attribute(&wsattrname).map_err(|e| format!("unable to open WSATTR: {}", e))?;

    let mr = state.cmaps.read().unwrap();
    let c2m = mr.get(&corpname).unwrap();

    let head_cid = if let Some(head_cid) = wslex.head2id(&head) {
        head_cid
    } else {
        return Err(format!("ERROR: '{}' not found in WSATTR lexicon", head));
    };
    let head_mid = c2m[head_cid as usize];
    if head_mid == u32::MAX {
        return Err(format!("ERROR: unable to translate '{}' with WSATTR id {} to model id'", head, head_cid));
    }
    let headx = if let Some(headx) = wmap.find_id(head_cid) {
        headx
    } else {
        return Err(format!("ERROR: model and lexicon know '{}', but it is not present in word sketch", head));
    };

    let mut out_rel: HashMap<String, HashMap<String, (usize, f64, f64)>> = std::collections::HashMap::new();

    for relx in headx.iter() {
        let rels = wslex.id2rel(relx.id);

        let out_coll: std::collections::HashMap<String, (usize, f64, f64)> = relx.iter().par_bridge().filter_map(|collx| {
            let mut lctx = Vec::<u32>::with_capacity(ntokens);
            let mut rctx = Vec::<u32>::with_capacity(ntokens);
            let colls = wslex.id2coll(collx.id);
            if collx.cnt < wsminfrq || collx.rnk < wsminrnk {
                return None;
            }

            let mut z = Array::<f64, Ix1>::zeros(vm.nmeanings());
            let mut zst = vec![];
            for _ in 0..vm.nmeanings() {
                zst.push(RunningStats::new());
            }
            for zk in zst.iter_mut() { (*zk).clear(); }
            let it = collx.iter();
            let mut rng = SmallRng::seed_from_u64(
                ((collx.id as u64) << 10) + (relx.id as u64));
            let itf = || -> Box<dyn Iterator<Item=(usize, Option<i32>)>> {
                if sampleconc > 0 && sampleconc < collx.cnt {
                    return Box::new(
                        it.sample(sampleconc as usize, &mut rng)
                    )
                }
                Box::new(it)
            };

            for (pos, _coll) in itf() {
                let _n_senses = expected_pi(&vm.counts, vm.alpha,
                                            head_mid, &mut z);

                for zk in z.iter_mut() {
                    if *zk < 1e-3 { *zk = 0.; }
                    *zk = zk.ln();  // ???
                }

                let start = if pos >= ntokens { pos - ntokens } else { 0 };
                let ctxit = wsattr.iter_ids(start as u64);

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
                    match c2m[*corpid as usize] {
                        u32::MAX => None,
                        id => Some(id),
                    }
                };

                for ctx_mid in lctx.iter()
                    .rev().filter_map(fmap_ids).take(window) {
                    var_update_z(&vm.in_vecs, &vm.out_vecs, &vm.code,
                                 &vm.path, head_mid, ctx_mid, &mut z);
                }

                for ctx_mid in rctx.iter()
                    .filter_map(fmap_ids).take(window) {
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

            let oc = if let Some(mp) = maxpos {
                // print!("\t{:.2}/{:.2}", zst[mp].mean(), zst[mp].stddev());
                (mp, zst[mp].mean(), zst[mp].stddev())
            } else {
                // print!("\t{:.2}/{:.2}", -1., -1.);
                (999, -1., -1.)
            };
            Some((colls.to_string(), oc))
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

    let config = rocket::Config { port: args.port, ..Default::default() };

    let models = std::sync::Arc::new(std::sync::RwLock::new(HashMap::new()));
    let modellist = std::fs::read_to_string(args.modelfile)?
        .lines()
        .filter_map(|line| {
            let stripped = line.trim();
            let parts = stripped.split('\t').collect::<Vec<_>>();
            if parts.len() != 2 {
                eprintln!("wrong number of elements while parsing modelfile: {}, should be 2", parts.len());
                eprintln!("got '{}'", stripped);
                return None;
            }
            Some((parts[0].to_string(), parts[1].to_string()))
        })
        .collect::<Vec<(String, String)>>();

    {
        let models = models.clone();
        tokio::spawn(async move {
            for (modellang, modelpath) in modellist {
                match load_model(&modellang, &modelpath, &models) {
                    Ok(_) => eprintln!("model {} for {} loaded", &modelpath, &modellang),
                    Err(e) => eprintln!("loading model {} for {} failed: {}", &modelpath, &modellang, &e),
                };
            }
        });
    }

    rocket::custom(config)
        .manage(ServerState{
            window:args.window, wsminfrq:args.wsminfrq, wsminrnk:args.wsminrnk, sampleconc: args.sampleconc, // defattr,
            cmaps: std::sync::RwLock::new(HashMap::new()), models: models.clone(),
    })
    .mount("/neighbors/", routes![req_neighbors])
    .mount("/wsdesamb/", routes![req_wsdesamb])
    .mount("/models/", routes![req_load_model, req_unload_model, req_list_models])
    .launch().await?;

    Ok(())
}

