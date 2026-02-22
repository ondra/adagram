use adagram::adagram::VectorModel;
use adagram::nn::nearest_mmul;
use adagram::nn::{nearest_mmul_dense_pool,build_dense_sense_pool};

use clap::Parser;

const VERSION: &str = git_version::git_version!(args=["--tags", "--always", "--dirty"]);

/// Retrieve nearest neighbors for a given word sense
#[derive(Parser, Debug)]
#[clap(author, version=VERSION, about)]
struct Args {
    /// adagram model path
    model: String,

    /// do not normalize vectors before querying neighbors
    #[clap(long, default_value_t = false)]
    no_norm: bool,

    /// minimum frequency of candidate neighbors
    #[clap(long, default_value_t = 5usize)]
    minfreq: usize,

    /// number of nearest neighbors to retrieve
    #[clap(long, default_value_t = 10usize)]
    neighbors: usize,

    /// minimum apriori sense probability for a sense to be reported
    #[clap(long, default_value_t = 1e-3f64)]
    minprob: f64,

    /// compact output: one line per sense with neighbors in a single space-separated column
    #[clap(long, default_value_t = false)]
    compact: bool,

    /// print informative progress messages to stderr
    #[clap(short, long, default_value_t = false)]
    verbose: bool,

    #[clap(long, default_value_t = false)]
    legacy: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // load model
    let (mut vm, id2str) = VectorModel::load_model(&args.model)?;

    if !args.no_norm {
        vm.norm();
    }
    let vm = vm;  // drop mut

    let dense_pool = if vm.normed && !args.legacy {
        if args.verbose {
            eprintln!("building dense pool (minfreq={})", args.minfreq);
        }
        let pool = build_dense_sense_pool(&vm, args.minfreq);
        if args.verbose {
            eprintln!("dense pool ready ({} senses)", pool.len());
        }
        Some(pool)
    } else {
        None
    };

    // build reverse lexicon mapping
    let mut str2id = std::collections::HashMap::<&str, u32>::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(word, id as u32);
    }

    if args.verbose {
        eprintln!("ready");
    }
    if args.compact {
        println!("hw\tsn\tnn\tminsim\tmaxsim");
    } else {
        println!("hw\tsn\tsim\tnn\tnn_sn");
    }
    for line in std::io::stdin().lines() {
        let unwrapped = line?;
        let mut parts = unwrapped.split_whitespace();
        let head = match parts.next() {
            Some(x) => x,
            None => { continue },
        };

        let head_id = match str2id.get(head) {
            Some(x) => *x,
            None => { eprintln!("Warning: skipping {} not in lexicon", head); continue },
        };

        let hvs = match dense_pool.as_ref() {
            Some(pool) => nearest_mmul_dense_pool(&vm, pool, head_id as usize, args.neighbors.saturating_add(1), args.minprob),
            None => nearest_mmul(&vm, head_id as usize, args.neighbors.saturating_add(1), args.minfreq, args.minprob),
        };
        for (sense, hv) in hvs.iter() {
            let filtered = hv.iter()
                .filter(|(i, _j, _sim)| *i as usize != head_id as usize)
                .take(args.neighbors)
                .collect::<Vec<_>>();
            if args.compact {
                let neighbors = filtered.iter()
                    .map(|(i, _j, _sim)| id2str[*i as usize].as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                let sim_upper = filtered.first().map(|(_, _, sim)| *sim).unwrap_or(0.0);
                let sim_lower = filtered.last().map(|(_, _, sim)| *sim).unwrap_or(0.0);
                println!("{}\t{}\t{}\t{}\t{}", head, sense, neighbors, sim_lower, sim_upper);
            } else {
                for (i, j, sim) in filtered {
                    println!("{}\t{}\t{}\t{}\t{}", head, sense, sim, id2str[*i as usize], j);
                }
            }
        }
    }

    Ok(())
}
