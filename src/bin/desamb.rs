use ndarray::Array;
use ndarray::prelude::*;

use adagram::adagram::VectorModel;
use adagram::common::*;

use clap::Parser;

const VERSION: &str = git_version::git_version!(args=["--tags","--always", "--dirty"]);

/// Assign concordance lines to senses
#[derive(Parser, Debug)]
#[clap(author, version=VERSION, about)]
struct Args {
    /// adagram model path
    model: String,

    /// ignore the headword during desambiguation
    #[clap(long,default_value_t=true)]
    skip_headword: bool,

    /// append the result to the input instead of outputting the classes only
    #[clap(long,default_value_t=true)]
    mirror_input: bool,

    // output probabilities for all senses
    #[clap(long,default_value_t=true)]
    print_probs: bool,

    // skip the first line of input
    #[clap(long,default_value_t=false)]
    skip_header: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if std::env::args().count() != 2 {
        return Err("usage: desamb MODELPATH".into());
    }

    // load model
    let modelpath = args.model;
    let (vm, id2str) = VectorModel::load_model(&modelpath)?;

    // build reverse lexicon mapping
    let mut str2id = std::collections::HashMap::<&str, u32>::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(word, id as u32);
    }

    let mut _headword_not_in_lexicon = 0;
    let mut _no_context = 0;
    let mut _all_invalid = 0;

    let mut z = Array::<f64, Ix1>::zeros(vm.nmeanings());
    let mut lines = std::io::stdin().lines().into_iter();
    if args.skip_header {
        if let Some(maybeline) = lines.next() {
            let fullline = maybeline?;
            let line = fullline.trim();
            if args.mirror_input {
                print!("{}\tcluster", line);
                if args.print_probs {
                    print!("\tcluster_probs");
                }
                println!();
            }
        }
    }
    for maybeline in lines {
        let fullline = maybeline?;
        let line = fullline.trim();
        let mut cols = line.split('\t');
        let head = match cols.next() {
            Some(x) => x,
            None => {
                eprintln!("=== EMPTY INPUT ===");
                continue
            },
        };

        let x = match str2id.get(head) {
            Some(n) => *n,
            None => {
                eprintln!("=== HEADWORD NOT IN LEXICON: {} ===", head);
                _headword_not_in_lexicon += 1;
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

        match cols.next() {
            None => {
                _no_context += 1;
            },
            Some(context) => {
                let parts = context.split_whitespace();
                for ctxword in parts {
                    if args.skip_headword && ctxword == head {
                        continue;
                    }
                    let y = match str2id.get(ctxword) {
                        Some(n) => { _nvalid += 1; *n },
                        None => { _ninvalid += 1; continue; },
                    };
                    var_update_z(&vm.in_vecs, &vm.out_vecs, &vm.code, &vm.path, x, y, &mut z);
                }
            },
        }

        if _nvalid < 1 {
            _all_invalid += 1;
        }

        exp_normalize(&mut z);

        let maxpos: Option<usize> = z.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i);

        if args.mirror_input {
            print!("{}", line);
        }

        if let Some(mp) = maxpos {
            print!("\t{}", mp);
        } else {
            print!("\t-1");
        }

        if args.print_probs {
            print!("\t{}", z.iter()
                     .enumerate()
                     .map(|(i, p): (usize, &f64)|->String {format!("{}:{:.2}", i, p)})
                     .collect::<Vec<String>>()  // won't be necessary in future rust
                     .join(" "));
        }
        println!();
    }

    eprintln!("encountered {} lines with no context", _no_context);
    eprintln!("encountered {} headwords not present in lexicon", _headword_not_in_lexicon);
    eprintln!("encountered {} lines where no context words were found in lexicon", _all_invalid);

    Ok(())
}

