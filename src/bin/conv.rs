use ndarray::prelude::*;
use adagram::adagram::VectorModel;
use adagram::common::expected_pi;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().count() != 3 {
        return Err("usage: conv SRCMODELPATH TGTMODELPATH".into());
    }

    // load model
    let srcmodelpath = std::env::args().nth(1).ok_or("source model path missing")?;
    let tgtmodelpath = std::env::args().nth(2).ok_or("target model path missing")?;
    if !tgtmodelpath.ends_with(".bvec32") {
        eprintln!("target model path does not end with .bvec32");
    }

    let mut bvecf = std::io::BufWriter::new(
        std::fs::File::create(&tgtmodelpath)?);
    let mut dicf = std::io::BufWriter::new(
        std::fs::File::create(tgtmodelpath.to_string() + ".dic")?);

    eprintln!("loading {}", srcmodelpath);
    let (mut vm, id2str) = VectorModel::load_model(&srcmodelpath)?;

    let norm = std::env::var("UNNORMED").is_err();
    if norm {
        eprintln!("normalizing vectors");
        vm.norm();
    }
    let vm = vm;  // drop mut

    let min_prob = 0.001;

    eprintln!("sorting by sense counts");
    let ii = vm.in_vecs.len_of(Axis(0));
    let jj = vm.in_vecs.len_of(Axis(1));
    let kk = vm.in_vecs.len_of(Axis(2));
    let mut indices = Vec::new();
    let mut z = Array::<f64, Ix1>::zeros(jj);
    for i in 0..ii {
        let _nsenses = expected_pi(&vm.counts, vm.alpha, i as u32, &mut z);
        for j in 0..jj {
            if z[j] < min_prob { continue; }
            indices.push((i, j)); 
        }
    }
    indices.sort_by_key(|&(i, j)| (vm.counts[[i, j]]) as usize);
    indices.reverse();

    eprintln!("writing {}", tgtmodelpath);
    writeln!(dicf, "{}", indices.len())?;
    writeln!(dicf, "{}", kk)?;
    for (i, j) in indices {
        writeln!(dicf, "{}##{}", id2str[i], j)?;
        for e in vm.in_vecs.slice(s![i, j, ..]).iter() {
            bvecf.write_all(&e.to_le_bytes())?;
        }
    }

    bvecf.flush()?;
    dicf.flush()?;

    eprintln!("done");
    Ok(())
}

