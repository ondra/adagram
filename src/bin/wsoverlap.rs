use clap::Parser;

use ndarray::Array;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::SmallRng;
use ndarray_rand::rand::SeedableRng;

use adagram::adagram::VectorModel;
use adagram::common::*;
use adagram::nn::nearest;
use adagram::reservoir_sampling::SamplerExt;

use corp::wsketch::WMap;
use corp::wsketch::WSLex;

use std::collections::{HashMap,HashSet};

const VERSION: &str = git_version::git_version!(args=["--tags","--always", "--dirty"]);

/// Assign Word Sketches to senses 
#[derive(Parser, Debug)]
#[clap(author, version=VERSION, about)]
struct Args {
    /// word sketch corpus
    corpname: String,

    /// attribute to use, should be compatible with the attribute used to train the model
    attrname: String,

    /// adaptive skip-gram model
    model: String,

    /// window size, token count on both sides of KWIC used for desambiguation
    #[clap(long,default_value_t=4)]
    window: usize,

    /// minimal apriori sense probability
    #[clap(long,default_value_t=1e-3)]
    sense_threshold: f64,

    /// amount of nearest neighbors for sense vectors to retrieve 
    #[clap(long,default_value_t=6)]
    sense_neighbors: usize,

    /// drop collocates below this rank
    #[clap(long,default_value_t=-100.)]
    wsminrnk: f32,

    /// drop collocates below this rank
    #[clap(long,default_value_t=1)]
    wsminfrq: u64,
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let corp = corp::corp::Corpus::open(&args.corpname)?;
    let attr: Box<dyn corp::corp::Attr> = corp.open_attribute(&args.attrname)?;

    let wsattrname = corp.get_conf("WSATTR").unwrap();
    let wsattr = corp.open_attribute(&wsattrname)?;
    let defattrname = corp.get_conf("DEFAULTATTR").unwrap();
    let defattr = corp.open_attribute(&defattrname)?;

    let wsbase = corp.get_conf("WSBASE").unwrap();
    let wmap = WMap::new(&wsbase)?;
    let wslex = WSLex::open(&wsbase, wsattr)?;

    eprintln!("ready");
    for line in std::io::stdin().lines() {
        let unwrapped = line?;
        let head = unwrapped.trim();
        
        let head_cid = if let Some(head_cid) = wslex.head2id(head) {
            head_cid
        } else {
            eprintln!("ERROR: '{}' not found in WSATTR lexicon", head);
            continue;
        };

        let headx = if let Some(headx) = wmap.find_id(head_cid) {
            headx
        } else {
            eprintln!("ERROR: model and lexicon know '{}', but it is not present in word sketch", head);
            continue;
        };

        eprintln!("processing {}", head);
        
        let mut hm = HashMap::<(u32, u32), HashSet<usize>>::new();
        //let mut hm = HashMap::new::<(u32, u32), HashSet<u64>>();

        for relx in headx.iter() {
            let rels = wslex.id2rel(relx.id);

            'coll: for collx in relx.iter() {
                let colls = wslex.id2coll(collx.id);
                //if collx.cnt < args.wsminfrq || collx.rnk < args.wsminrnk {
                //    continue 'coll;
                //}
                let poss = hm.entry((relx.id, collx.id))
                    .or_insert_with(|| HashSet::<usize>::new());
                poss.extend(collx.iter().map(|(pos, _coll)| pos));

                //print!("{}\t{}\t{}\t{}\t{}\t",
                //       head, rels, colls, collx.cnt, collx.rnk);

                /*
                let maxpos: Option<usize> = zst.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.mean().total_cmp(&b.mean()))
                    .map(|(i, _)| i);

                if let Some(mp) = maxpos {
                    if zst[mp].mean() < args.clusterminprob
                            || zst[mp].n() < args.clusterminfrq {
                        print!("\t-");
                    } else {
                        print!("\t{}", mp);
                    }
                    print!("\t{:.2}\t{:.2}", zst[mp].mean(), zst[mp].stddev());
                } else {
                    print!("\t-\t-\t-");
                }

                print!("\t{}", zst.iter()
                    .enumerate()
                    .map(|(i, p): (usize, &RunningStats)| -> String {
                        format!("{}:{:.2}/{:.2}", i, p.mean(), p.stddev())})
                    .collect::<Vec<String>>()
                    .join(" "));
                println!();
                */
            }
        }

    }

    Ok(())
}

#[derive(Debug)]
struct RunningStats {
    m_n: usize,
    m_oldm: f64,
    m_newm: f64,
    m_olds: f64,
    m_news: f64,
}

impl RunningStats {
    fn new() -> RunningStats {
        RunningStats { m_n: 0, m_oldm: 0., m_newm: 0., m_olds: 0., m_news: 0. }
    }
    fn clear(&mut self) {
        self.m_n = 0;
    }

    fn push(&mut self, x: f64) {
        self.m_n += 1;

        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if self.m_n == 1 {
            self.m_oldm = x;
            self.m_newm = x;
            self.m_olds = 0.0;
        } else {
            self.m_newm = self.m_oldm + (x - self.m_oldm)/self.m_n as f64;
            self.m_news = self.m_olds + (x - self.m_oldm)*(x - self.m_newm);

            // set up for next iteration
            self.m_oldm = self.m_newm;
            self.m_olds = self.m_news;
        }
    }

    fn n(&self) -> usize { self.m_n }
    fn mean(&self) -> f64 {
        if self.m_n >= 1 { self.m_newm } else { 0.0 }
    }
    fn var(&self) -> f64 {
        if self.m_n > 1 { self.m_news/(self.m_n - 1) as f64 } else { 0.0 }
    }
    fn stddev(&self) -> f64 {
        self.var().sqrt()
    }
}
