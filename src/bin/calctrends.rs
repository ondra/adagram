use clap::Parser;

use adagram::diachronic::*;
use slope::*;

const VERSION: &str = git_version::git_version!(args=["--tags", "--always", "--dirty"]);

/// Calculate diachronic trend for all words in a corpus
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

    /// minimum norm for a structure attribute value to be considered
    #[clap(long,default_value_t=0.15)]
    epoch_limit: f64,

    /// prefix to which to write the output files
    #[clap(long)]
    trendbase: String,
}

use std::time::{Instant,Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let corpus = corp::corp::Corpus::open(&args.corpname)?;

    eprintln!("opening attribute {}", &args.posattr);
    let posattr = corpus.open_attribute(&args.posattr)?;
    eprintln!("opening attribute {}", &args.diaattr);
    let diaattrparts = &mut args.diaattr.split('.');
    let diastructname = diaattrparts.next().ok_or("")?;
    let diastructattr = corpus.open_attribute(&args.diaattr)?;
    let diastruct = corpus.open_struct(diastructname)?;

    let trendbase = if args.trendbase == "" {
        corpus.path + "/" + &args.diaattr + "." + &args.posattr
    } else { args.trendbase };
    eprintln!("writing output to {}", trendbase);

    let (diamap, new_norms, _ordered_epochnames) =
        map_diavals(diastructattr.as_ref(), args.epoch_limit)?;

    let h = posattr.id_range() as usize;
    let w = new_norms.len();
    eprintln!("w {}", w);

    if w < 2 {
        eprintln!("WARNING: less than 2 valid structattr values");
        return Err("semantic error".into());
    }

    if h <= 0 {
        eprintln!("WARNING: empty corpus");
        return Err("semantic error".into());
    }
    
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
    let mut mktf = std::fs::File::create(trendbase.clone() + ".mkts_all.trends")?;
    let mut mktwr = BinaryTrendsWriter::new(&mut mktf)?;
    let mut mkmf = std::fs::File::create(trendbase.clone() + ".mkts_all.minigraphs")?;
    let mut mkmwr = BinaryMinigraphWriter::new(&mut mkmf)?;

    let mut lrtf = std::fs::File::create(trendbase.clone() + ".linreg_all.trends")?;
    let mut lrtwr = BinaryTrendsWriter::new(&mut lrtf)?;
    let mut lrmf = std::fs::File::create(trendbase.clone() + ".linreg_all.minigraphs")?;
    let mut lrmwr = BinaryMinigraphWriter::new(&mut lrmf)?;

    let mut normed = vec![0.0f64; w];
    let mut rel = vec![0.0f64; w];
    let mut next_report_time = Instant::now() + Duration::from_secs(60);
    let xs = (0..w).map(|x| x as f64).collect::<Vec<_>>();
    for attr_id in 0..posattr.id_range() as usize {
        if attr_id & 0xfff == 0 && Instant::now() >= next_report_time {
            eprintln!("visited {} ids out of {} ({:.2} %)",
                attr_id, h, 100.*(attr_id as f64)/(h as f64));
            next_report_time += Duration::from_secs(240);
        }

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

        let (lp, lslope) = linreg(&xs[..], &rel);
        let (mp, mslope) = mk(&xs[..], &rel);

        let lslope_over_all_periods = lslope * (w as f64 - 1.);
        let mslope_over_all_periods = mslope * (w as f64 - 1.);

        let langle = lslope_over_all_periods.atan() * (180./3.141592);
        let mangle = mslope_over_all_periods.atan() * (180./3.141592);

        mktwr.put(attr_id as u32, mangle.round() as i8, mp as f32)?;
        lrtwr.put(attr_id as u32, langle.round() as i8, lp as f32)?;

        let mut resampled = vec![0f64; 8];
        resample(&rel[..], &mut resampled[..]);
        let decimated = resampled.into_iter().map(|n| (n*6.0) as u8).collect::<Vec::<u8>>();
        mkmwr.put(attr_id as u32, &decimated)?;
        lrmwr.put(attr_id as u32, &decimated)?;
    }

    use std::io::Write;
    mktf.flush()?;
    mkmf.flush()?;
    lrtf.flush()?;
    lrmf.flush()?;
    mktf.sync_all()?;
    mkmf.sync_all()?;
    lrtf.sync_all()?;
    lrmf.sync_all()?;

    eprintln!("done.");
    Ok(())
}

