use ndarray::Array;
use ndarray::prelude::*;

use adagram::adagram::VectorModel;
use adagram::common::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().count() != 2 {
        return Err("usage: desamb MODELPATH".into());
    }

    // load model
    let modelpath = std::env::args().nth(1).ok_or("model path missing")?;
    let (vm, id2str) = VectorModel::load_model(&modelpath)?;

    // build reverse lexicon mapping
    let mut str2id = std::collections::HashMap::<&str, u32>::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(word, id as u32);
    }

    let mut z = Array::<f64, Ix1>::zeros(vm.nmeanings());
    for line in std::io::stdin().lines() {
        let unwrapped = line?;
        let mut parts = unwrapped.split_whitespace();
        let head = match parts.next() {
            Some(x) => x,
            None => {
                println!("=== EMPTY INPUT ===");
                continue
            },
        };
        
        let x = match str2id.get(head) {
            Some(n) => *n,
            None => {
                println!("=== HEADWORD NOT IN LEXICON: {} ===", head);
                continue;
            },
        };

        let _n_senses = expected_pi(&vm.counts, vm.alpha, x, &mut z);

        for zk in z.iter_mut() {
            if *zk < 1e-3 { *zk = 0.; }
            *zk = zk.ln();  // ???
        }

        let mut _nvalid = 0;
        let mut _ninvalid = 0;
        for ctxword in parts {
            let y = match str2id.get(ctxword) {
                Some(n) => { _nvalid += 1; *n },
                None => { _ninvalid += 1; continue; },
            };
            var_update_z(&vm.in_vecs, &vm.out_vecs, &vm.code, &vm.path, x, y, &mut z);
        }

        // if nvalid < 1;

        exp_normalize(&mut z);

        let maxpos: Option<usize> = z.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i);

        if let Some(mp) = maxpos {
            print!("\t{}", mp);
        } else {
            print!("\t-");
        }

        println!("{}", z.iter()
                 .enumerate()
                 .map(|(i, p): (usize, &f64)|->String {format!("{}:{:.2}", i, p)})
                 .collect::<Vec<String>>()  // won't be necessary in future rust
                 .join(" "));
    }

    Ok(())
}

