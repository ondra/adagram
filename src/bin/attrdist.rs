use clap::Parser;

const VERSION: &str = git_version::git_version!(args=["--tags", "--always", "--dirty"]);

/// Train an adaptive skip-gram model
#[derive(Parser, Debug)]
#[clap(author, version=VERSION, about)]
struct Args {
    /// source corpus
    corpname: String,

    /// positional attributes
    attrnames: Vec<String>,

    /// number of training threads to run in parallel
    #[clap(long,default_value_t=1)]
    threads: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let corp = corp::corp::Corpus::open(&args.corpname)?;

    let attrs = args.attrnames.iter().map(
        |attrname| -> Result<Box<dyn corp::corp::Attr + Sync + Send>, Box<dyn std::error::Error>> {
            Ok(corp.open_attribute(attrname)?)
        })
        .collect::<Result<Vec<Box<dyn corp::corp::Attr + Sync + Send>>, Box<dyn std::error::Error>>>()?;
    let total_words = attrs[0].text().size();
    // eprintln!("{} positions in total", total_words);
    let _total_words = total_words;

    let words_read = std::sync::atomic::AtomicUsize::new(0);
    let starttime = std::time::Instant::now();

    let trainfunc = | thread_id: usize, | -> std::collections::HashMap<Vec<u32> ,u64> {
        let mut words_read_last = 0;
        let mut reporttime = std::time::Instant::now();

        let startpos = thread_id * total_words / args.threads;
        let endpos = (thread_id + 1) * total_words / args.threads;
        let partsize = endpos - startpos;

        let starttime = std::time::Instant::now();

        let mut hm = std::collections::HashMap::new();
        let mut its = attrs.iter()
            .map(|attr| attr.iter_ids(startpos as u64).take(endpos - 1))
            .collect::<Vec<_>>();
        for pos in 0..partsize {
            if pos & 0xffff == 0xffff {
                let local_words_read = words_read.fetch_add(0xffff, std::sync::atomic::Ordering::Relaxed);
                if thread_id == 0 {
                    let dur = reporttime.elapsed().as_secs_f64();
                    if dur > 0.5 {
                        let rws = local_words_read - words_read_last;
                        words_read_last = local_words_read;
                        let wps = rws as f64 / dur;
                        let remaining_words = total_words - local_words_read;
                        let remaining_secs = if wps != 0.0 { remaining_words / wps as usize } else { 0 };
                        let remaining_hours = remaining_secs / 3600;
                        let remaining_mins = (remaining_secs % 3600) / 60;
                        reporttime = std::time::Instant::now();
                        let elapsed = reporttime.checked_duration_since(starttime).map(|d| d.as_secs()).unwrap_or(0);
                        eprintln!("\r[{}] visited {} positions out of {} ({:.2} %), {:.0} wps, {:02}h:{:02}m remaining", elapsed,
                            local_words_read, total_words, local_words_read as f64 / total_words as f64 * 100.0,
                            wps, remaining_hours, remaining_mins,
                        );
                    }
                }
            }
             
            let out = its.iter_mut().map(|it| it.next().unwrap_or(0)).collect::<Vec<u32>>();
            *hm.entry(out).or_insert(0) += 1;
        }
        hm
    };

    let res = if args.threads > 1 {
        let hms = std::thread::scope(|scope| {
            let mut handles = vec![];
            for thread_id in 0..args.threads {
                let thread_id_c = thread_id;
                handles.push(
                    scope.spawn(move ||
                        trainfunc(thread_id_c)
                    )
                );
            }

            handles.into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        eprintln!("corpus positions visited, collecting output");
        let mut ohm = std::collections::HashMap::<Vec<u32>, u64>::new();
        for hm in hms {
            for (key, value) in hm {
                *ohm.entry(key).or_insert(0) += value;
            }
        }
        ohm
    } else {
        trainfunc(0)
    };

    eprintln!("read {} words, {} wps",
              total_words, 
              total_words as f32 / starttime.elapsed().as_secs() as f32);

    for (key, value) in res {
        for (attr, k) in std::iter::zip(attrs.iter(), key) {
            print!("{}\t", attr.id2str(k));
        }
        println!("{}", value);
    }

    eprintln!("done");
    Ok(())
}

