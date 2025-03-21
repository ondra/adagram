use ndarray_rand::rand::prelude::SmallRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::rand_distr::Beta;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Binomial;
use ndarray_rand::rand_distr::Distribution;

pub struct ReservoirSampler<'a, E> {
    rng: &'a mut SmallRng,
    elems: Vec<E>,
    sample_size: usize,
}

impl <E> ReservoirSampler<'_, E> {
    pub fn new(sample_size: usize, rng: &'_ mut SmallRng) -> ReservoirSampler<'_, E> {
        ReservoirSampler { elems: Vec::<E>::with_capacity(sample_size),
            rng, sample_size,
        }
    }
    pub fn push(&mut self, item: E) {
        if self.elems.len() < self.sample_size {
            self.elems.push(item);
        } else {
            let dist = Uniform::from(0..self.sample_size);
            let r = dist.sample(self.rng);
            self.elems[r] = item;
        }
    }
    pub fn clear(&mut self) { self.elems.clear(); }
    pub fn elems(&self) -> &[E] { self.elems.as_ref() }
}

pub struct ReservoirSamplerG<'a, E: Default, const N: usize> {
    rng: &'a mut SmallRng,
    elems: [E; N],
    c: usize,
}

impl <'a, E: Default, const N: usize> ReservoirSamplerG<'_, E, { N }> {
    pub fn new(rng: &'_ mut SmallRng) -> ReservoirSamplerG<'_, E, { N }> {
        ReservoirSamplerG { elems: core::array::from_fn(|_i| E::default()), rng, c: 0 }
    }
    pub fn push(&mut self, item: E) {
        if self.c < N {
            self.elems[self.c] = item;
            self.c += 1;
        } else {
            let dist = Uniform::from(0..N);
            let r = dist.sample(self.rng);
            self.elems[r] = item;
        }
    }
    pub fn clear(&mut self) { self.c = 0; }
    pub fn elems(&self) -> &[E] { &self.elems[0..self.c] }
}



pub struct Sampler<'a, I, R>
    where R: Rng
{
    it: I,
    n: isize,
    k: usize,
    rng: &'a mut R,
}

impl<I, R: Rng> Sampler<'_, I, R>
    where I: ExactSizeIterator
{
    fn from_iter(it: I, k: usize, rng: &'_ mut R) -> Sampler<'_, I, R> {
        let n = it.len() as isize;
        Sampler { it, n, k, rng }
    }

}

// Wikipedia says that to draw a beta-binomial random variate X ~ BetaBin(n, alpha, beta), simply
// draw a p ~ Beta(alpha, beta) and then draw X ~ B(n, p)
// https://en.wikipedia.org/wiki/Beta-binomial_distribution
//
// https://engineering.tableau.com/fun-with-optimal-random-sampling-b08d72e77ade#87eb
//def in_order_sampler(n, k, offset=0):
//    if n < 0 or k <= 0:
//        return
//    nskip = distributions.betabinom.rvs(n, 1, k)
//    yield offset + nskip
//    yield from in_order_sampler(n-nskip-1, k-1, offset+nskip+1)
//

// ^ that code seems to be wrong, https://research.tableau.com/sites/default/files/OptimalSRS.pdf as reproduced below works

fn _beta_binom(alpha: f64, beta: f64, n: u64, rng: &mut impl Rng) -> u64 {
    let d1 = Beta::new(alpha, beta).expect("Beta distribution");
    let p = d1.sample(rng);
    let d2 = Binomial::new(n, p).expect("Binomial distribution"); 
    d2.sample(rng)
}

fn beta_binom1(beta: f64, n: u64, rng: &mut impl Rng) -> u64 {
    let d1 = Uniform::new(0., 1.);
    let s: f64 = d1.sample(rng);
    let p = 1. - s.powf(1./beta);
    let d2 = Binomial::new(n, p).expect("Binomial distribution"); 
    d2.sample(rng)
}

impl<T, I: Iterator + Iterator<Item = T>, R> Iterator for Sampler<'_, I, R>
    where R: Rng, //+ ?Sized
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.n < 0 || self.k == 0 {
            None
        } else {
            let nskip = beta_binom1(
            // let nskip = beta_binom(1.,
                self.k as f64, (self.n as usize - self.k) as u64, self.rng);
            for _ in 0..nskip {
                self.it.next();
            }
            self.n = self.n - nskip as isize - 1;
            self.k -= 1;
            self.it.next()
        }
    }
}

pub trait SamplerExt<'a, T, R: Rng>: ExactSizeIterator<Item = T> + Sized 
{
    fn sample(self, k: usize, rng: &'a mut R) -> Sampler<'a, Self, R> {
        Sampler::from_iter(self, k, rng)
    }
}

impl <T, I: ExactSizeIterator<Item = T>, R: Rng> SamplerExt<'_, T, R> for I
{} 
