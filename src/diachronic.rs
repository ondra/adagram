//fn create_tag_ordering(keys: Vec<&str>) -> Vec<(String, Vec<KT>)> {

//}

// re.split(r'\D+')

type DigitGroup = Vec<u32>;
type DigitGroups = Vec<DigitGroup>;

fn digit_groups(s: &str) -> DigitGroups {
    s.chars().collect::<Vec<_>>()
        .split(|c| !c.is_ascii_digit())
        .map(|numseq| numseq.iter()
             .map(|d| d.to_digit(10).unwrap())
             .skip_while(|&d| d == 0)
             .collect())
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

fn create_tag_ordering(keys: Vec<&str>) -> Vec<(usize, String, DigitGroups)>
{
    let mut x = keys.iter()
        .enumerate()
        .map(|(i, s)| 
             (i, s.to_string(), digit_groups(s)))
        .collect::<Vec<_>>();
    x.sort_by(|(_, _, k1), (_, _, k2)| cmp_keys(k1, k2));
    x
}

fn all_slopes(xs: Vec<f64>, ys: Vec<f64>) -> Vec<f64> {
    assert!(xs.len() == ys.len());
    let n = xs.len();
    let mut xsd_triu = Vec::new();
    let mut ysd_triu = Vec::new();
    for i in 0..n {
        for j in 0..i {
            xsd_triu.push(xs[i]-xs[j]);
            ysd_triu.push(ys[i]-ys[j]);
        }
    }

    let mut slopes = std::iter::zip(ysd_triu.iter(), xsd_triu.iter())
        .map(|(x, y)| x / y)
        .collect::<Vec<_>>();

    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_slope = match slopes.len() % 2 {
        1 => slopes[slopes.len() / 2] + slopes[slopes.len() / 2 + 1] / 2.,
        0 => slopes[slopes.len() / 2],
        _ => unreachable!(),
    };

    let s: f64 = std::iter::zip(xsd_triu.iter(), ysd_triu.iter())
        .map(|(x, y)|
             (x * y).signum())
        .sum();

    // group ys by value
    let mut ycs = std::collections::HashMap::<BitHashedF64, usize>::new();
    for y in ys { *ycs.entry(BitHashedF64{0: 0.}).or_insert(0) += 1; }

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

    vec![]
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
    #[test]
    fn test_cmp_numeric() {
        assert!(cmp_numeric(&vec![1], &vec![2]).is_lt());
        assert!(cmp_numeric(&vec![2], &vec![11]).is_lt());
        assert!(cmp_numeric(&vec![0,0,2], &vec![11]).is_lt());
    }
}

