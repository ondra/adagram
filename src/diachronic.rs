#![allow(dead_code)]
#![allow(unused)]

type DigitGroups = Vec<usize>;

fn digit_groups(s: &str) -> DigitGroups {
    s.chars().collect::<Vec<_>>()
        .split(|c| !c.is_ascii_digit())
        .map(|numseq| numseq.iter()
             .map(|d| d.to_digit(10).unwrap())
             .skip_while(|&d| d == 0)
             .fold(0usize, |accum, digit| accum * 10 + digit as usize)
             )
        .collect()
}

fn cmp_keys(a: &DigitGroups, b: &DigitGroups) -> std::cmp::Ordering {
    let mut ia = a.iter();
    let mut ib = b.iter();
    loop {
        match (ia.next(), ib.next()) {
            (None, None) => { return std::cmp::Ordering::Equal; },
            (None, Some(_bv)) => { return std::cmp::Ordering::Less; },
            (Some(_av), None) => { return std::cmp::Ordering::Greater; },
            (Some( av), Some(bv)) => {
                if av == bv { continue; }
                else if av > bv { return std::cmp::Ordering::Greater; }
                else { return std::cmp::Ordering::Less; }
            }
        }
    }
}

/*
fn cmp_numeric(a: &DigitGroup, b: &DigitGroup) -> std::cmp::Ordering {
    let ia = a.iter().skip_while(|&&d| d == 0);
    let ib = b.iter().skip_while(|&&d| d == 0);
    let iac = ia.clone().count();
    let ibc = ib.clone().count();
    if      iac > ibc { return std::cmp::Ordering::Greater; }
    else if iac < ibc { return std::cmp::Ordering::Less; }
    for (av, bv) in std::iter::zip(ia, ib) {
        if      av > bv { return std::cmp::Ordering::Greater; }
        else if av < bv { return std::cmp::Ordering::Less; }
    }
    std::cmp::Ordering::Equal
}
*/

pub fn tag_ordering<'a>(keys: &'a Vec<&'a str>) -> Vec<(usize, &'a str, DigitGroups)>
{
    let mut x = keys.iter()
        .enumerate()
        .map(|(i, s)| 
             (i, *s, digit_groups(s)))
        .collect::<Vec<_>>();
    x.sort_by(|(_, _, k1), (_, _, k2)| cmp_keys(k1, k2));
    x
}

fn clamp(n: u8, max: u8) -> u8 { if n > max { max } else { n } }

pub struct BinaryTrendsWriter<'a> { dest: &'a mut dyn std::io::Write, }
impl BinaryTrendsWriter<'_> {
    pub fn new(dest: &'_ mut dyn std::io::Write) -> Result<BinaryTrendsWriter<'_>, Box<dyn std::error::Error>> {
        let mut header = [0u8; 32];
        for (i, e) in b"manatee trends v1.1".iter().enumerate() {
            header[i] = *e;
        }
        dest.write_all(&header)?;
        Ok(BinaryTrendsWriter { dest })
    }
    pub fn put(&mut self, id: u32, slope: i8, p: f32) -> Result<(), Box<dyn std::error::Error>> {
        self.dest.write_all(&id.to_le_bytes())?;
        self.dest.write_all(&slope.to_le_bytes())?;
        self.dest.write_all(&p.to_le_bytes())?;
        Ok(())
    }
}

pub struct BinaryMinigraphWriter<'a> { dest: &'a mut dyn std::io::Write, }
impl BinaryMinigraphWriter<'_> {
    pub fn new(dest: &'_ mut dyn std::io::Write) -> Result<BinaryMinigraphWriter<'_>, Box<dyn std::error::Error>> {
        let mut header = [0u8; 32];
        for (i, e) in b"manatee minigraphs v1.1".iter().enumerate() {
            header[i] = *e;
        }
        dest.write_all(&header)?;
        Ok(BinaryMinigraphWriter { dest })
    }
    pub fn put(&mut self, id: u32, nums: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        self.dest.write_all(&id.to_le_bytes())?;
        let v: u32 = nums.into_iter().enumerate().map(|(i, &n)| (clamp(n, 15) as u32) << (i * 4)).sum();
        self.dest.write_all(&v.to_le_bytes())?;
        Ok(())
    }
}

pub fn resample(from: &[f64], to: &mut [f64]) {
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

pub fn map_diavals(diastructattr: &dyn corp::corp::Attr, epoch_limit: f64)
        -> Result<(Vec<u32>, Vec<f64>, Vec<String>), Box<dyn std::error::Error>> {
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

    let avg_norm = total_norm as f64 / epoch_count as f64;
    eprintln!("average norm is {}", avg_norm);
    let min_norm = if epoch_limit >= 1. { epoch_limit as u64 }
    else {
        let adj_min_norm = (avg_norm * epoch_limit) as u64;
        eprintln!("adjusted EPOCH_LIMIT is {}", adj_min_norm);
        adj_min_norm
    };

    fn format_key(key: &Vec<usize>) -> String {
        "<".to_string() + &key.iter().map(|v| format!("{}", v))
            .collect::<Vec<_>>().join(", ") + ">"
    }

    let mut newid = 0;
    let mut new_norms = Vec::<f64>::new();
    let mut ordered_epochnames = Vec::<String>::new();
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
            ordered_epochnames.push(s.to_string());
        }
    }
    Ok((diamap, new_norms, ordered_epochnames))
}
