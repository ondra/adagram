use ndarray::prelude::*;

use adagram::adagram::VectorModel;
use adagram::nn::nearest;

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
        str2id.insert(word, id as u32);
    }

    for line in std::io::stdin().lines() {
        let unwrapped = line?;
        let mut parts = unwrapped.split_whitespace();
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

        let hv = nearest(&vm, head_id as usize, senseno, 10, 5);
        for (i, j, sim) in hv.iter() {
            println!("{} {} {}", sim, id2str[*i as usize], j);
        }
    }

    Ok(())
}

