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

    use medians::Medianf64;
    let median_slope = slopes.as_slice().medf_checked().unwrap();

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
            let t = b / rad.sqrt();
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

