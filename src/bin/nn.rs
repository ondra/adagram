use ndarray::Array;
use ndarray::prelude::*;

use adagram::adagram::VectorModel;

use binary_heap_plus::BinaryHeap;

fn norm_l2<S, D>(a: &ArrayBase<S, D>) -> f32
    where
    S: ndarray::Data<Elem = f32>,
    D: ndarray::Dimension
{
    //a.iter().map(|x| x.square()).sum::<f64>().sqrt()
    a.iter().map(|x| *x**x).sum::<f32>().sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().count() != 2 {
        return Err("usage: nn MODELPATH".into());
    }

    // load model
    let modelpath = std::env::args().nth(1).ok_or("model path missing")?;
    let (vm, id2str) = VectorModel::load_model(&modelpath)?;

    // build reverse lexicon mapping
    let mut str2id = std::collections::HashMap::<&str, u32>::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(&word, id as u32);
    }

    let ii = vm.in_vecs.len_of(Axis(0));
    let jj = vm.in_vecs.len_of(Axis(1));
    let mut sims = Array::<f32, Ix2>::zeros((ii, jj));

    for line in std::io::stdin().lines() {
        let unwrapped = line?;
        let mut parts = unwrapped.trim().split_whitespace();
        let head = match parts.next() {
            Some(x) => x,
            None => { println!("=== EMPTY INPUT ==="); continue },
        };

        let head_id = match str2id.get(head) {
            Some(x) => *x,
            None => { println!("=== HEAD NOT IN LEXICON ==="); continue },
        };

        let senseno_s = match parts.next() {
            Some(x) => x,
            None => { println!("=== MISSING SENSE NO ==="); continue },
        };

        let senseno = match senseno_s.parse::<usize>() {
            Ok(y) => y,
            Err(e) => { println!("=== WRONG SENSE NO: {} ===", e); continue },
        };
        
        if senseno >= vm.in_vecs.len_of(Axis(1)) {
            println!("=== INVALID SENSE NO ==="); continue
        }


        // sims.fill(0.);
        let qvec_r = vm.in_vecs.slice(s![head_id as usize, senseno, ..]);
        let n = norm_l2(&qvec_r);
        let qvec = qvec_r.mapv(|v| v / n) ;

        let min_count = 5.;

        for i in 0..ii { for j in 0..jj {
            sims[[i, j]] = if vm.counts[[i, j]] < min_count { -f32::INFINITY }
            else {
                let v = vm.in_vecs.slice(s![i, j, ..]);
                qvec.dot(&v) / norm_l2(&v)
            }
        }}

        let top_k = 10;

        let sf = |(id1, s1): &(u32, u32), (id2, s2): &(u32, u32)| {
            let sim1 = sims[[*id1 as usize, *s1 as usize]];
            let sim2 = sims[[*id2 as usize, *s2 as usize]];
            sim2.partial_cmp(&sim1).unwrap_or_else(
                || match (sim2.is_nan(), sim1.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (false, true) => std::cmp::Ordering::Greater,
                    (true, false) => std::cmp::Ordering::Less,
                    (false, false) => panic!(),
                }
            )
        };

        let mut heap = BinaryHeap::new_by(sf);

        for i in 0..ii { for j in 0..jj {
            heap.push((i as u32, j as u32)); 
            if heap.len() > top_k { heap.pop(); }
        }}

        let mut hv: Vec<_> = heap.drain().collect();
        hv.sort_by(sf);
        for (i, j) in hv.iter() {
            let sim = sims[[*i as usize, *j as usize]];
            println!("{} {} {}", sim, id2str[*i as usize], j);
        }
    }

    Ok(())
}

