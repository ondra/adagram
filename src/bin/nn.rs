use adagram::adagram::VectorModel;
use adagram::nn::nearest_mmul;

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

    /// compact output: one line per sense with neighbors in a single space-separated column
    #[clap(long, default_value_t = false)]
    compact: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // load model
    let (mut vm, id2str) = VectorModel::load_model(&args.model)?;

    if !args.no_norm {
        vm.norm();
    }
    let vm = vm;  // drop mut

    // build reverse lexicon mapping
    let mut str2id = std::collections::HashMap::<&str, u32>::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(word, id as u32);
    }

    eprintln!("ready");
    if args.compact {
        println!("hw\tnn\tminsim\tmaxsim");
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

        let hvs = nearest_mmul(&vm, head_id as usize, args.neighbors.saturating_add(1), args.minfreq);
        for (sense, hv) in hvs.iter().enumerate() {
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
                println!("{}\t{}\t{}\t{}", head, neighbors, sim_lower, sim_upper);
            } else {
                for (i, j, sim) in filtered {
                    println!("{}\t{}\t{}\t{}\t{}", head, sense, sim, id2str[*i as usize], j);
                }
            }
        }
    }

    Ok(())
}
