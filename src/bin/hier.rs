use clap::Parser;

use ndarray::prelude::*;

use adagram::adagram::VectorModel;
use adagram::nn::sim;
use adagram::nn::nearest;

const VERSION: &str = git_version::git_version!(args=["--tags","--always", "--dirty"]);

/// Induce hierarchy on senses
#[derive(Parser, Debug)]
#[clap(author, version=VERSION, about)]
struct Args {
//    /// word sketch corpus
//    corpname: String,

//    /// attribute to use, should be compatible with the attribute used to train the model
//    attrname: String,

    /// adaptive skip-gram model
    model: String,

//    /// window size, token count on both sides of KWIC used for desambiguation
//    #[clap(long,default_value_t=4)]
//    window: usize,

    /// minimal apriori sense probability
    #[clap(long,default_value_t=1e-3)]
    sense_threshold: f64,

    /// amount of nearest neighbors for sense vectors to retrieve 
    #[clap(long,default_value_t=6)]
    sense_neighbors: usize,
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    eprintln!("loading model");
    let (vm, id2str) = VectorModel::load_model(&args.model)?;

    eprintln!("inverting model lexicon");
    let mut str2id = std::collections::HashMap::<&str, u32>
            ::with_capacity(id2str.len());
    for (id, word) in id2str.iter().enumerate() {
        str2id.insert(word, id as u32);
    }

    eprintln!("ready");

    for line in std::io::stdin().lines() {
        let unwrapped = line?;
        let head = unwrapped.trim();

        let head_mid = if let Some(id) = str2id.get(head) {
            *id as usize
        } else {
            eprintln!("ERROR: '{}' not found in WSATTR lexicon", head);
            continue;
        };

        let nsenses = vm.in_vecs.len_of(Axis(1));

        let min_cnt = 5.;

        let mut neighbors = Vec::<Vec<(u32, u32, f32)>>::new();

        for i in 0..nsenses { 
            let r = nearest(&vm, head_mid as usize, i,
                            args.sense_neighbors, 5);
            print!("# sense {} ({}):", i, vm.counts[[head_mid as usize, i]]);
            for (mid, senseno, sim) in &r {
                if *mid as usize == head_mid && *senseno as usize == i { continue; }
                print!("\t{}##{}/{:.3}", id2str[*mid as usize], senseno, sim);
            }
            neighbors.push(r);
            println!();
        }
        println!();

        for i in 0..nsenses { 
            if vm.counts[[head_mid, i]] < min_cnt { continue; }
            print!("\t#{}", i);
        }
        println!();

        for i in 0..nsenses { 
            if vm.counts[[head_mid, i]] < min_cnt { continue; }
            print!("{}", i);
            for j in 0..nsenses { 
                if vm.counts[[head_mid, j]] < min_cnt { continue; }
                if i == j { print!("\t - "); }
                else { print!("\t{:.3}", sim(&vm, head_mid, i, head_mid, j)); }
            }
            println!();
        }

        let mut nodes = Vec::new();
        for i in 0..nsenses { 
            if vm.counts[[head_mid, i]] < min_cnt { continue; }
            nodes.push(Node::Leaf(i));
        }

        while nodes.len() > 1 {
            let mut maxs = -2.;
            let mut i1 = 0;
            let mut i2 = 0;

            for i in 0..nodes.len() {
                for j in 0..i {
                    let s = node_sim(&vm, head_mid, &nodes[i], &nodes[j]);

                    if s > maxs {
                        maxs = s;
                        i1 = i;
                        i2 = j;
                    }
                }
            }

            let n1 = nodes.remove(i1);
            let n2 = nodes.remove(i2);

            nodes.push(Node::Internal(
                Box::new(n1), Box::new(n2)
            ));
        }

        let root = &nodes[0];

        let sf = |senseno: usize| -> String {
            let mut out = "".to_string();
            for (mid, n_senseno, sim) in neighbors[senseno].iter() {
                if *mid as usize == head_mid { continue; }
                out.push_str(&format!(" {}##{}/{:.3}", id2str[*mid as usize], n_senseno, sim));
            }
            out
        };

        fn fmt_node<F>(node: &Node, vm: &VectorModel, head_mid: usize, sf: &F) -> String
            where F: Fn(usize) -> String
        {
            match node {
                Node::Leaf(ix) => {
                    format!("#{} {}\n", ix, sf(*ix))
                },
                Node::Internal(l, r) =>
                    format!("{:.3}d\n{}{}", node_diameter(&vm, head_mid, node),
                            &indent(&fmt_node(l, vm, head_mid, sf)),
                            &indent(&fmt_node(r, vm, head_mid, sf)),
                    )
            }
        }

        println!();
        println!("{}", fmt_node(&root, &vm, head_mid, &sf));
    }

    Ok(())
}

enum Node {
    Leaf(usize),
    Internal(Box<Node>, Box<Node>),
}

fn node_sim(vm: &VectorModel, head_mid: usize, a: &Node, b: &Node) -> f32 {
    let mut cd = -2.;
    for u in a.ids() {
        for v in b.ids() {
            let s = sim(&vm, head_mid, u, head_mid, v);
            if s > cd {
                cd = s;
            }
        }
    }
    cd
}

fn node_diameter(vm: &VectorModel, head_mid: usize, n: &Node) -> f32 {
    let ids = n.ids();
    let mut smin = 2.;
    for u in ids.iter() {
        for v in ids.iter() {
            let s = sim(&vm, head_mid, *u, head_mid, *v);
            if s < smin {
                smin = s;
            }
        }
    }
    smin
}

fn indent(s: &str) -> String {
    let mut out = String::new();
    for part in s.lines() {
        if part != "" {
            out.push_str("    ");
            out.push_str(part);
            out.push_str("\n");
        } else {
            out.push_str("\n");
        }
    }
    out
}

impl Node {
    fn ids(&self) -> Vec<usize> {
        let mut v = Vec::new();
        self.ids_(&mut v);
        v
    }
    fn ids_(&self, v: &mut Vec<usize>) {
        match self {
            Self::Leaf(id) => { v.push(*id); },
            Self::Internal(l, r) => { l.ids_(v); r.ids_(v); }
        }
    }
}

