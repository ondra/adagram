use clap::Parser;

use adagram::diachronic::*;

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

    /// text type structure attribute
    typeattr: String,
    
    #[clap(long,default_value_t=5)]
    minfreq: usize,

    /// minimum norm for a structure attribute value to be considered
    #[clap(long,default_value_t=0.15)]
    epoch_limit: f64,

    /// number of worker threads to use (0 to use all processors)
    #[clap(long, default_value_t=0)]
    nthreads: usize,

    /// skip positions with a specific attribute value (specify as attribute:value1,2,3,...)
    #[clap(long)]
    skip: Option<String>,
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
    eprintln!("opening attribute {}", &args.diaattr);
    let diaattrparts = &mut args.diaattr.split('.');
    let diastructname = diaattrparts.next().ok_or("")?;
    let diastructattr = corpus.open_attribute(&args.diaattr)?;
    let diastruct = corpus.open_struct(diastructname)?;

    let typeattrparts = &mut args.typeattr.split('.');
    let typestructname = typeattrparts.next().ok_or("")?;
    let typestructattr = corpus.open_attribute(&args.typeattr)?;
    let typestruct = corpus.open_struct(typestructname)?;

    let typenorms = typestructattr.get_freq("token:l")?;

    let ntype = typestructattr.id_range();

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

    let (diamap, new_norms) = map_diavals(diastructattr.as_ref(), args.epoch_limit)?;

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

    use std::sync::{Arc,Mutex};
    let sense_diacnts = Arc::new(Mutex::new(vec![0f64; ntype as usize * epochcnt]));

    eprintln!("ready");
    if true { // tsv
        println!("hw\ttt\tnorm\ttotal");
    }
    for line in std::io::stdin().lines() {
        {
            let mut sense_diacnts = sense_diacnts.lock().unwrap();
            sense_diacnts.iter_mut().for_each(|v| *v = 0.);
        }
        let unwrapped = line?;
        let head = unwrapped.trim();

        let corpid = if let Some(cid) = posattr.str2id(head) {
            cid
        } else {
            eprintln!("ERROR: {} not found in corpus lexicon", head);
            continue;
        };

        let poss = posattr.revidx().id2poss(corpid);
        let pit = poss.par_bridge().map_init(
            || {

            },
            |(), pos|{
            let diastructpos = if let Some(diastructpos) = diastruct.num_at_pos(pos) {
                diastructpos
            } else {
                eprintln!("WARN: position {} for id {} is outside structure", pos, corpid);
                return None;
            };

            let diaid = diastructattr.text().get(diastructpos);
            let epoch_no = diamap[diaid as usize];
            if epoch_no == u32::MAX {
                return None;
            }

            if let Some((skipattr, skipids)) = &skip {
                let curskipattrid = skipattr.text().get(diastructpos);
                if skipids.contains(&curskipattrid) {
                    return None;
                }
            }

            let typestructpos = if let Some(typestructpos) = typestruct.num_at_pos(pos) {
                typestructpos
            } else {
                eprintln!("WARN: position {} for id {} is outside structure", pos, corpid);
                return None;
            };

            let typeid = typestructattr.text().get(typestructpos);

            // let ctxit = posattr.iter_ids(start);

            Some((typeid, epoch_no))
        }).flatten();
        pit.for_each(|(typeid, epoch_no)|{
            let mut sense_diacnts = sense_diacnts.lock().unwrap();
            sense_diacnts[typeid as usize*epochcnt + epoch_no as usize] += 1.;
        });
        let sense_diacnts = sense_diacnts.lock().unwrap();

        let freqs = (0..epochcnt)
            .map(|epoch|
                (0..ntype).map(|type_| sense_diacnts[type_ as usize*epochcnt + epoch]).sum()
            ).collect::<Vec<f64>>();

        let types = (0..ntype)
            .map(|type_|
                (0..epochcnt).map(|epoch| sense_diacnts[type_ as usize*epochcnt + epoch]).sum()
            ).collect::<Vec<f64>>();

        if false { // print block format
            println!("HW {}", head);
            for type_ in 0..ntype as usize {
                print!("t##{}", type_);
                print!("\t{}", typenorms.frq(type_ as u32));
                print!("\t{}", types[type_]);
                for epoch in 0..epochcnt {
                    print!("\t{}", sense_diacnts[type_ as usize*epochcnt + epoch]);
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
        } else { // print tsv, only texttype data
            for type_ in 0..ntype as usize {
                println!("{}\t{}\t{}\t{}", head, type_, typenorms.frq(type_ as u32), types[type_]);
            }
        }

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

