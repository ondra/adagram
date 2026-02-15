use ndarray::Array;
use ndarray::prelude::*;

use adagram::adagram::VectorModel;
use adagram::common::*;

use clap::Parser;

const VERSION: &str = git_version::git_version!(args=["--tags", "--always", "--dirty"]);

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

    /// minimum apriori probability for a sense to be taken into consideration
    #[clap(long, default_value_t=1e-3f64)]
    min_prob: f64,

    /// use uniform prior probabilities for senses
    #[clap(long, default_value_t=false)]
    uniform_prob: bool,

    /// print informative progress messages to stderr
    #[clap(short, long, default_value_t = false)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.verbose {
        eprintln!("loading {}", args.model);
    }
    let (vm, id2str) = VectorModel::load_model(&args.model)?;

    if args.verbose {
        eprintln!("building str2id");
    }
    let mut str2id = std::collections::HashMap::<&str, u32>::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(word, id as u32);
    }
    if args.verbose {
        eprintln!("done");
    }

    let mut headword_not_in_lexicon = 0;
    let mut ctx_empty = 0;
    let mut ctx_all_unknown = 0;
    let mut allok = 0;

    let mut z = Array::<f64, Ix1>::zeros(vm.nmeanings());
    let mut lines = std::io::stdin().lines().into_iter();
    let mut colnames = if args.skip_header {
        if let Some(maybeline) = lines.next() {
            let fullline = maybeline?;
            let line = fullline.trim_end_matches(|c| { c == '\n' || c == '\r' || c == ' '});
            line.split("\t").take(args.skip_columns).map(|v| v.to_string()).collect::<Vec<_>>()
        } else { vec![] }
    } else { vec![] };

    colnames.resize(args.skip_columns, "".to_string());
    for colno in 0..args.skip_columns {
        if colnames[colno] == "" {
            colnames[colno] = format!("col{}", colno);
        }
    }
    colnames.push("head".to_string());
    colnames.push("context".to_string());
    if args.print_status { colnames.push("status".to_string()); }
    colnames.push("cluster".to_string());
    if args.print_probs { colnames.push("cluster_probs".to_string()) };
    if args.print_nsenses { colnames.push("nsenses".to_string()) };

    if args.print_header {
        for colname in colnames { print!("{}\t", colname); }
        println!();
    }

    let emit = |vs: Vec<_>| {
        println!("{}", vs.join("\t"));
    };

    for maybeline in lines {
        let mut outvals = vec![];

        let fullline = maybeline?;
        let line = fullline.trim_end_matches(|c| { c == '\n' || c == '\r' || c == ' '});
        let mut cols = line.split('\t');

        for _ in 0..args.skip_columns {
            let cv = cols.next();
            if args.mirror_input {
                if let Some(t) = cv { outvals.push(t.to_string()) }
                else { outvals.push("".to_string()) }
            }
        }

        let head = match cols.next() {
            Some(x) => {
                if args.mirror_input { outvals.push(x.to_string()) }
                x
            },
            None => {
                if args.mirror_input {
                    outvals.push("".to_string());
                    outvals.push("".to_string());
                }
                if args.print_status {
                    outvals.push("empty".to_string());
                }
                emit(outvals);
                continue
            },
        };

        let x = match str2id.get(head) {
            Some(n) => {
                *n
            },
            None => {
                // eprintln!("=== HEADWORD NOT IN LEXICON: {} ===", head);
                if args.mirror_input {
                    outvals.push("".to_string());
                }
                if args.print_status {
                    outvals.push("nhead".to_string());
                }
                emit(outvals);
                headword_not_in_lexicon += 1;
                continue;
            },
        };

        let n_senses = expected_pi(&vm.counts, vm.alpha, x, &mut z, args.min_prob);

        if args.uniform_prob {
            for zk in z.iter_mut() {
                if *zk < args.min_prob { *zk = 0.; }
                else { *zk = 1. / n_senses as f64; }
            }
        } else {
            for zk in z.iter_mut() {
                if *zk < args.min_prob { *zk = 0.; }
                *zk = zk.ln();  // ???
            }
        }

        let mut nvalid = 0;
        let mut ninvalid = 0;

        match cols.next() {
            Some(context) => {
                let parts = context.split_whitespace();
                for ctxword in parts {
                    if args.skip_headword && ctxword == head {
                        continue;
                    }
                    let y = match str2id.get(ctxword) {
                        Some(n) => { nvalid += 1; *n },
                        None => { ninvalid += 1; continue; },
                    };
                    var_update_z(&vm.in_vecs, &vm.out_vecs, &vm.code, &vm.path, x, y, &mut z);
                }
                if args.mirror_input { outvals.push(context.to_string()) }
            },
            None => {
                if args.mirror_input { outvals.push("".to_string()) }
            },
        }

        if nvalid == 0 {
            if ninvalid == 0 {
                ctx_empty += 1;
                if args.print_status { outvals.push("noctx".to_string()); }
            } else {
                ctx_all_unknown += 1;
                if args.print_status { outvals.push("unctx".to_string()); }
            }
        } else {
            allok += 1;
            if args.print_status { outvals.push("allok".to_string()); }
        }

        exp_normalize(&mut z);

        let maxpos: Option<usize> = z.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i);

        if let Some(mp) = maxpos { outvals.push(format!("c{}", mp)) }
        else { outvals.push("-1".to_string()) }

        if args.print_probs {
            outvals.push(format!("{}", z.iter()
                     .enumerate()
                     .map(|(i, p): (usize, &f64)|->String {format!("{}:{:.2}", i, p)})
                     .collect::<Vec<String>>()  // won't be necessary in future rust
                     .join(" ")));
        }
        if args.print_nsenses { outvals.push(format!("{}", n_senses)) }
        emit(outvals);
    }

    if headword_not_in_lexicon > 0 || ctx_empty > 0 || ctx_all_unknown > 0 || allok == 0 {
        eprintln!("encountered {} headwords not present in lexicon", headword_not_in_lexicon);
        eprintln!("encountered {} lines where no context words were present", ctx_empty);
        eprintln!("encountered {} lines where no context words were found in lexicon", ctx_all_unknown);
    }

    Ok(())
}
