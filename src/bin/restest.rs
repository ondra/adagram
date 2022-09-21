
use adagram::reservoir_sampling::SamplerExt;
use ndarray_rand::rand::thread_rng;

fn main() {
    let mut tr = thread_rng();

    let m = 50;
    let r = 1000000;

    let mut v = Vec::new();
    for _i in 0..m {
        v.push(0u64);
    }

    for _j in 0..r {
        for i in (0..m).into_iter().sample(5, &mut tr) {
            v[i] += 1;
            // println!("{}", i);
        }
    }

    for (i, e) in v.iter().enumerate() {
        println!("{}\t{}\t{}", i, *e, *e as f64 / r as f64);
    }
}
