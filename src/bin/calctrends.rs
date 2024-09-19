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

    let diavals: Vec<_> = (0..diastructattr.id_range())
        .map(|did| diastructattr.id2str(did))
        .collect();

    // eprintln!("diavals: {:?}", diavals);
    let bad_values = std::collections::HashSet::from([
        "===NONE===",
        "===NONE==",
        "===NONE=",
        "===NONE",
        "===NON",
        "===NO",
        "===N",
        "9999",
        "",
    ]);

    let to = tag_ordering(&diavals);
    let mut diamap = [u32::MAX].into_iter().cycle().take(diastructattr.id_range() as usize)
        .collect::<Vec<u32>>();
    let norms = diastructattr.get_freq("token:l")?;
    let mut epoch_count = 0;
    let mut total_norm = 0;
    for (orgid, s, _key) in &to {
        if !bad_values.contains(s) {
            epoch_count += 1;
            total_norm += norms.frq(*orgid as u32);
        }
    }

    let min_norm = args.epoch_limit;
    let avg_norm = total_norm as f64 / epoch_count as f64;
    eprintln!("average norm is {}", avg_norm);
    let min_norm = if min_norm >= 1. { min_norm as u64 }
    else {
        let adj_min_norm = (avg_norm * min_norm) as u64;
        eprintln!("adjusted EPOCH_LIMIT is {}", adj_min_norm);
        adj_min_norm
    };

    fn format_key(key: &Vec<usize>) -> String {
        "<".to_string() + &key.iter().map(|v| format!("{}", v))
            .collect::<Vec<_>>().join(", ") + ">"
    }

    let mut newid = 0;
    let mut new_norms = Vec::<f64>::new();
    eprintln!("values ordered as (index, value, norm, original index, key):");
    for (orgid, s, key) in &to {
        let norm = norms.frq(*orgid as u32);
        if bad_values.contains(s) || norm < min_norm {
            eprintln!("\t-\t{}\t{}\t{}\t{}",
                s, norm, *orgid, format_key(&key));
        } else {
            eprintln!("\t{}\t{}\t{}\t{}\t{}",
                newid, s, norm, *orgid, format_key(&key));
            diamap[*orgid] = newid;
            newid += 1;
            new_norms.push(norm as f64);
        }
    }

    let h = posattr.id_range() as usize;
    let w = newid as usize;
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

        let (lp, lslope) = adagram::diachronic::linreg(&xs[..], &rel);
        let (mp, mslope) = adagram::diachronic::mk(&xs[..], &rel);

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

struct BinaryTrendsWriter<'a> { dest: &'a mut dyn std::io::Write, }
impl BinaryTrendsWriter<'_> {
    fn new(dest: &'_ mut dyn std::io::Write) -> Result<BinaryTrendsWriter<'_>, Box<dyn std::error::Error>> {
        let mut header = [0u8; 32];
        for (i, e) in b"manatee trends v1.1".iter().enumerate() {
            header[i] = *e;
        }
        dest.write_all(&header)?;
        Ok(BinaryTrendsWriter { dest })
    }
    fn put(&mut self, id: u32, slope: i8, p: f32) -> Result<(), Box<dyn std::error::Error>> {
        self.dest.write_all(&id.to_le_bytes())?;
        self.dest.write_all(&slope.to_le_bytes())?;
        self.dest.write_all(&p.to_le_bytes())?;
        Ok(())
    }
}


fn clamp(n: u8, max: u8) -> u8 { if n > max { max } else { n } }

struct BinaryMinigraphWriter<'a> { dest: &'a mut dyn std::io::Write, }
impl BinaryMinigraphWriter<'_> {
    fn new(dest: &'_ mut dyn std::io::Write) -> Result<BinaryMinigraphWriter<'_>, Box<dyn std::error::Error>> {
        let mut header = [0u8; 32];
        for (i, e) in b"manatee minigraphs v1.1".iter().enumerate() {
            header[i] = *e;
        }
        dest.write_all(&header)?;
        Ok(BinaryMinigraphWriter { dest })
    }
    fn put(&mut self, id: u32, nums: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        self.dest.write_all(&id.to_le_bytes())?;
        let v: u32 = nums.into_iter().enumerate().map(|(i, &n)| (clamp(n, 15) as u32) << (i * 4)).sum();
        self.dest.write_all(&v.to_le_bytes())?;
        Ok(())
    }
}

fn resample(from: &[f64], to: &mut [f64]) {
    let frompart = 1.0f64 / from.len() as f64;
    let topart = 1.0f64 / to.len() as f64;
    if to.len() > from.len() {
        for i in 0..to.len() {
            let frombeg = ((frompart * (i as f64)) / topart) as usize;
            let fromend = ((frompart * (i as f64) + 1.0) / topart) as usize;
            for j in frombeg..fromend {
                to[i] += from[j] / (fromend - frombeg) as f64;
            }
        }
    } else {
        for i in 0..to.len() {
            let topos = i as f64 * topart;
            let j = (topos / frompart) as usize;
            to[i] = from[j];
        }
    }
}


