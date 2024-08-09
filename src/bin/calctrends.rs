use clap::Parser;

use adagram::diachronic::*;


const VERSION: &str = git_version::git_version!(args=["--tags","--always", "--dirty"]);

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
}

use std::time::{Instant,Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let corpus = corp::corp::Corpus::open(&args.corpname)?;

    eprintln!("opening attribute {}", &args.posattr);
    let posattr = corpus.open_attribute(&args.posattr)?;
    eprintln!("opening attribute {}", &args.diaattr);
    // let diaattr = corpus.open_attribute(&args.diaattr)?;
    let diaattrparts = &mut args.diaattr.split('.');
    let diastructname = diaattrparts.next().ok_or("")?;
    let diastructattr = corpus.open_attribute(&args.diaattr)?;
    
    // let diapath = corpus.path.clone() + &args.diaattr;
    // let dialex = corp::lex::MapLex::open(&diapath)?;
    // let diatext = corp::text::Int::open(&diapath)?;
    let diastruct = corpus.open_struct(diastructname)?;

    // let diaattr = corpus.open_attribute(&args.diaattr)?;
    let n = diastructattr.id_range();


    let diavals: Vec<_> = (0..diastructattr.id_range())
        .map(|did| diastructattr.id2str(did))
        .collect();

    let mut total_norm = 0u64;
    let mut epoch_count = 0u64;

    // eprintln!("diavals: {:?}", diavals);

    let to = tag_ordering(&diavals);
    // eprintln!("to: {:?}", diavals);

    let h = posattr.id_range() as usize;
    let w = diavals.len() as usize;
    eprintln!("dl {}", diavals.len());
    eprintln!("w {}", w);

    if w < 2 {
        eprintln!("WARNING: less than 2 valid structattr values");
        return Err("semantic error".into());
    }

    if h <= 0 {
        eprintln!("WARNING: empty corpus");
        return Err("semantic error".into());
    }

    let diamap = (0..w as u32).collect::<Vec<u32>>();
    
    eprintln!("using {} diachronic attribute ids and {} ids ({} MB)",
              w, h, 8 * w * h / 1024 / 1024);
    let mut freqs = vec![0f64; w * h];

    let mut diaiditer = diastructattr.iter_ids(0);
    let mut structpos = 0u64;
    let mut diaid = diaiditer.next().unwrap();
    let mut structbeg = diastruct.beg_at(structpos) as usize;
    let mut structend = diastruct.end_at(structpos) as usize;

    let mut next_report_time = Instant::now() + Duration::from_secs(60);
    let text_size = posattr.text().size();
    for (pos, attr_id) in std::iter::zip(
            0..text_size,
            posattr.text().posat(0).unwrap()) {
        if pos & 0xfff == 0 && Instant::now() >= next_report_time {
            eprintln!("visited {} positions out of {} ({:.2} %)",
                pos, h, 100.*(pos as f64)/(text_size as f64));
            next_report_time += Duration::from_secs(240);
        }
        while pos >= structend {
            structpos += 1;
            diaid = diaiditer.next().unwrap();
            structbeg = diastruct.beg_at(structpos) as usize;
            structend = diastruct.end_at(structpos) as usize;
        }
        if pos < structbeg {
            continue;
        }

        let epoch_no = diamap[diaid as usize];
        if epoch_no == u32::MAX { continue; }
        freqs[attr_id as usize * w + epoch_no as usize] += 1.;
    }

    eprintln!("frequencies collected, calculating trends");

    let mut next_report_time = Instant::now() + Duration::from_secs(60);
    let xs = (0..w).map(|x| x as f64).collect::<Vec<_>>();
    for attr_id in 0..posattr.id_range() as usize {
        if attr_id & 0xfff == 0 && Instant::now() >= next_report_time {
            eprintln!("visited {} ids out of {} ({:.2} %)",
                attr_id, h, 100.*(attr_id as f64)/(h as f64));
            next_report_time += Duration::from_secs(240);
        }

        let (p, slope) = adagram::diachronic::mk(
            &xs[..], &freqs[attr_id..attr_id+w]);

        let (mp, mslope) = adagram::diachronic::mk(
            &xs[..], &freqs[attr_id..attr_id+w]);

    }

    eprintln!("done.");
    Ok(())
}

















