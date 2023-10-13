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

    /// output probabilities for all senses
    #[clap(long,default_value_t=true)]
    print_probs: bool,

    /// output the total number of apriori senses
    #[clap(long,default_value_t=true)]
    print_nsenses: bool,

    /// skip the first line of input
    #[clap(long)]
    skip_header: bool,

    /// write column names as the first output line
    #[clap(long)]
    print_header: bool,

    /// number of tab-separated columns to skip from the left for processing
    #[clap(long, default_value_t=0)]
    skip_columns: usize,

    /// emit column with desambiguation status
    #[clap(long, default_value_t=true)]
    print_status: bool,

    /// minimum probability for a sense to be taken into consideration
    #[clap(long, default_value_t=1e-3f64)]
    min_prob: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("loading {}", args.model);
    let (vm, id2str) = VectorModel::load_model(&args.model)?;

    eprintln!("building str2id");
    let mut str2id = std::collections::HashMap::<&str, u32>::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(word, id as u32);
    }
    eprintln!("done");

    let mut _headword_not_in_lexicon = 0;
    let mut _no_context = 0;
    let mut _all_invalid = 0;

    let mut z = Array::<f64, Ix1>::zeros(vm.nmeanings());
    let mut lines = std::io::stdin().lines().into_iter();
    let colnames = if args.skip_header {
        if let Some(maybeline) = lines.next() {
            let fullline = maybeline?;
            let line = fullline.trim_end_matches(|c| { c == '\n' || c == '\r' || c == ' '});
            line.split("\t").take(args.skip_columns).map(|v| v.to_string()).collect::<Vec<_>>()
        } else { vec![] }
    } else { vec![] };

    if args.print_header {
        for colname in colnames { print!("{}\t", colname); }
        print!("head\tcontext");
        if args.print_status { print!("\tstatus"); }
        print!("\tcluster");
        if args.print_probs { print!("\tcluster_probs") };
        if args.print_nsenses { print!("\tnsenses") };
        println!();
    }

    for maybeline in lines {
        let fullline = maybeline?;
        let line = fullline.trim_end_matches(|c| { c == '\n' || c == '\r' || c == ' '});
        let mut cols = line.split('\t');

        for _ in 0..args.skip_columns {
            let colv = match cols.next() {
                Some(col) => { col },
                None => { "" },
            };
            if args.mirror_input {
                print!("{}\t", colv);
            }
        }

        let head = match cols.next() {
            Some(x) => x,
            None => {
                // eprintln!("=== EMPTY INPUT ===");
                if args.print_status {
                    print!("empty");
                }
                println!();
                continue
            },
        };

        let x = match str2id.get(head) {
            Some(n) => {
                if args.mirror_input { print!("{}\t", n) }
                *n
            },
            None => {
                // eprintln!("=== HEADWORD NOT IN LEXICON: {} ===", head);
                if args.mirror_input {
                    print!("\t\t");
                }
                if args.print_status {
                    print!("nhead");
                }
                println!();
                _headword_not_in_lexicon += 1;
                continue;
            },
        };

        let n_senses = expected_pi(&vm.counts, vm.alpha, x, &mut z, args.min_prob);

        for zk in z.iter_mut() {
            if *zk < args.min_prob { *zk = 0.; }
            *zk = zk.ln();  // ???
        }

        let mut _nvalid = 0;
        let mut _ninvalid = 0;

        match cols.next() {
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
                if args.mirror_input { print!("{}\t", context) }
            },
            None => {
                _no_context += 1;
                if args.mirror_input { print!("\t") }
            },
        }

        if _nvalid < 1 {
            _all_invalid += 1;
            if args.print_status {
                print!("noctx");
            } else {
                print!("allok");
            }
        }

        exp_normalize(&mut z);

        let maxpos: Option<usize> = z.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i);

        print!("\t");

        if let Some(mp) = maxpos { print!("{}", mp); }
        else { print!("-1"); }

        if args.print_probs {
            print!("\t{}", z.iter()
                     .enumerate()
                     .map(|(i, p): (usize, &f64)|->String {format!("{}:{:.2}", i, p)})
                     .collect::<Vec<String>>()  // won't be necessary in future rust
                     .join(" "));
        }
        if args.print_nsenses { print!("\t{}", n_senses); }
        println!();
    }

    eprintln!("encountered {} lines with no context", _no_context);
    eprintln!("encountered {} headwords not present in lexicon", _headword_not_in_lexicon);
    eprintln!("encountered {} lines where no context words were found in lexicon", _all_invalid);

    Ok(())
}

