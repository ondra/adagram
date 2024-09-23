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

/// calculate the Theil-Sen and Mann-Kendall statistics
pub fn mk(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    assert!(xs.len() == ys.len());
    let n = xs.len();
    let triulen = (n*(n+1))/2;
    let mut slopes = Vec::<f64>::with_capacity(triulen);
    let mut s = 0f64;
    for i in 0..n {
        for j in 0..i {
            let x = xs[i] - xs[j];
            let y = ys[i] - ys[j];
            slopes.push(y / x);
            let v = x * y;
            if v != 0. { s += v.signum(); }
        }
    }

    let l = slopes.len();
    let median_slope = if l % 2 == 0 {
        (*order_stat::kth_by(&mut slopes, l / 2 - 1, |u, v| u.partial_cmp(v).unwrap()) +
        *order_stat::kth_by(&mut slopes, l / 2, |u, v| u.partial_cmp(v).unwrap())) / 2.
    } else {
        *order_stat::kth_by(&mut slopes, l / 2, |u, v| u.partial_cmp(v).unwrap())
    };

    // group ys by value
    let mut ycs = std::collections::HashMap::<BitHashedF64, usize>::new();
    for y in ys { *ycs.entry(BitHashedF64{0: *y}).or_insert(0) += 1; }

    // extract sizes of groups of same-valued ys larger than 1
    let ties = ycs.into_iter()
        .filter_map(|(v, c)| {
            if c > 1 { Some(c) }
            else { None }
        }).collect::<Vec<_>>();
    // group the groups by their sizes
    let mut group_sizes = std::collections::HashMap::<usize, usize>::new();
    for group_size in ties {
        *group_sizes.entry(group_size).or_insert(0) += 1;
    }

    let adj: usize = group_sizes.iter().map(|(group_size, count)| {
        group_size * count * (count-1) * (2*count+5)
    }).sum();

    let v = n * (n-1) * (2*n+5) - adj;
    let v = v as f64/18.;

    let z = if s > 0. {
        (s-1.) / v.sqrt()
    } else if s < 0. {
        (s+1.) / v.sqrt()
    } else {
        0.
    };

    let standard_normal = statrs::distribution::Normal::new(0., 1.).unwrap();
    use statrs::distribution::ContinuousCDF;
    let p = 2.*(1.-standard_normal.cdf(z.abs()));

    (p, median_slope)
}

fn mean(vs: &[f64]) -> f64 { vs.iter().sum::<f64>() / vs.len() as f64 }

/// Calculate the Simple Linear Regression p-value and slope
pub fn linreg(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    assert!(xs.len() == ys.len());
    let n = xs.len();
    assert!(n >= 2);

    let meany = mean(&ys);
    let meanx = mean(&xs);

    let btop = std::iter::zip(xs.iter(), ys.iter())
        .map(|(x, y)| (y - meany)*(x - meanx))
        .sum::<f64>();
    let bbot = xs.iter()
        .map(|x| (x - meanx) * (x - meanx))
        .sum::<f64>();
    let b = btop / bbot;

    let a = meany - b*meanx;

    let evaluated = xs.iter()
        .map(|x| b*x + a)
        .collect::<Vec<f64>>();
    let residuals = std::iter::zip(evaluated.iter(), ys.iter())
        .map(|(e, y)| e - y)
        .collect::<Vec<f64>>();

    let p = if n == 2 {
        0.
    } else {
        let var = residuals.iter()
            .map(|r| r*r)
            .sum::<f64>()
            /
            (n - 2) as f64;
        let rad = var /
            xs.iter()
            .map(|x| (x - meanx)*(x - meanx))
            .sum::<f64>();

        if rad <= 0. {
            1.
        } else {
            let rads = rad.sqrt();
            let t = if rads >= 0. { b / rads }
                else { 1000. };
            let tdist = statrs::distribution::StudentsT::new(0., 1., (n - 2) as f64).unwrap();
            use statrs::distribution::ContinuousCDF;
            2. * (1. - tdist.cdf(t.abs()))
        }
    };
    (p, b)
}

struct BitHashedF64 (f64);
impl std::hash::Hash for BitHashedF64 {
    fn hash<H>(&self, state: &mut H)
        where H: std::hash::Hasher
    { self.0.to_bits().hash(state); }
}
impl std::cmp::PartialEq for BitHashedF64 {
    fn eq(&self, other: &Self) -> bool { self.0.to_bits() == other.0.to_bits() }
}
impl std::cmp::Eq for BitHashedF64 {}


#[cfg(test)]
mod tests {
    use crate::diachronic::*;
    /*#[test]
    fn test_cmp_numeric() {
        assert!(cmp_numeric(&vec![1], &vec![2]).is_lt());
        assert!(cmp_numeric(&vec![2], &vec![11]).is_lt());
        assert!(cmp_numeric(&vec![0,0,2], &vec![11]).is_lt());
    }*/

    #[test]
    fn test_linreg() {
        assert_eq!(linreg(&[0.,1.,2.,3.,4.], &[0., 2., 4., 6., 8.]), (1.0, 2.0));
        assert_eq!(linreg(&[0.,1.,2.,3.,4.], &[1., 2., 3., 4., 5.]), (1.0, 1.0));
        assert_eq!(linreg(&[0.,1.,2.,3.,4.], &[1., 1., 1., 1., 1.]), (1.0, 0.0));
    }

    #[test]
    fn test_mk() {
        assert_eq!(mk(&[0.,1.,2.,3.,4.], &[1., 1., 1., 1., 1.]), (1.0, 0.0));
        assert_eq!(mk(&[0.,1.,2.,3.,4.], &[1., 2., 3., 4., 5.]), (0.027486336110310372, 1.0));
        assert_eq!(mk(&[0.,1.,2.,3.,4.], &[0., 2., 4., 6., 8.]), (0.027486336110310372, 2.0));
    }

    #[test]
    fn test_tag_ordering() {
        let tags = vec!["2020-008", "2022-01", "2019", "1982-04-02", "2018-02"];
        let ord = tag_ordering(&tags);
        assert_eq!(ord,
            [(3, "1982-04-02", vec![1982, 4, 2]),
             (4, "2018-02",    vec![2018, 2]),
             (2, "2019",       vec![2019]),
             (0, "2020-008",   vec![2020, 8]),
             (1, "2022-01",    vec![2022, 1])]);
    }
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
        -> Result<(Vec<u32>, Vec<f64>), Box<dyn std::error::Error>> {
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
    Ok((diamap, new_norms))
}
