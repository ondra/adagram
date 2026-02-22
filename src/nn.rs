use binary_heap_plus::BinaryHeap;
use ndarray::prelude::*;
use crate::adagram::VectorModel; 
use crate::common::expected_pi;

pub struct DenseSensePool {
    dim: usize,
    ids: Vec<(u32, u32)>,
    vecs: Vec<f32>,
}

impl DenseSensePool {
    pub fn len(&self) -> usize { self.ids.len() }
    pub fn is_empty(&self) -> bool { self.ids.is_empty() }
}

fn norm_l2<S, D>(a: &ArrayBase<S, D>) -> f32
    where
    S: ndarray::Data<Elem = f32>,
    D: ndarray::Dimension
{
    a.iter().map(|x| *x**x).sum::<f32>().sqrt()
}

impl VectorModel {
    /// make all input vectors unit length
    pub fn norm(&mut self) {
        let ii = self.in_vecs.len_of(Axis(0));
        let jj = self.in_vecs.len_of(Axis(1));
        for i in 0..ii {
            for j in 0..jj {
                let mut v = self.in_vecs.slice_mut(s![i, j, ..]);
                let n = norm_l2(&v);
                v.iter_mut().for_each(|e| *e /= n );
            }
        }
        self.normed = true;
    }
}

pub fn sim(vm: &VectorModel,
            head_id1: usize, senseno1: usize,
            head_id2: usize, senseno2: usize) -> f32
{
    let qvec_r1 = vm.in_vecs.slice(s![head_id1, senseno1, ..]);
    let qnorm1 = norm_l2(&qvec_r1);
    let qvec1 = qvec_r1.mapv(|v| v / qnorm1);

    let qvec_r2 = vm.in_vecs.slice(s![head_id2, senseno2, ..]);
    let qnorm2 = norm_l2(&qvec_r2);
    let qvec2 = qvec_r2.mapv(|v| v / qnorm2);

    qvec1.dot(&qvec2)
}

pub fn sim_normed(vm: &VectorModel,
            head_id1: usize, senseno1: usize,
            head_id2: usize, senseno2: usize) -> f32
{
    let qvec1 = vm.in_vecs.slice(s![head_id1, senseno1, ..]);
    let qvec2 = vm.in_vecs.slice(s![head_id2, senseno2, ..]);
    qvec1.dot(&qvec2)
}

pub fn nearest(vm: &VectorModel, head_id: usize, senseno: usize, top_k: usize, min_count: usize) -> Vec<(u32, u32, f32)>
{
    let ii = vm.in_vecs.len_of(Axis(0));
    let jj = vm.in_vecs.len_of(Axis(1));

    let qvec_r = vm.in_vecs.slice(s![head_id, senseno, ..]);
    let qnorm = norm_l2(&qvec_r);
    let qvec = qvec_r.mapv(|v| v / qnorm);

    let sf = |(_id1, _s1, sim1): &(u32, u32, f32), (_id2, _s2, sim2): &(u32, u32, f32)| {
        sim2.partial_cmp(sim1).unwrap_or_else(
            || match (sim2.is_nan(), sim1.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (false, true) => std::cmp::Ordering::Greater,
                (true, false) => std::cmp::Ordering::Less,
                (false, false) => panic!(),
            }
        )
    };

    let mut heap = BinaryHeap::new_by(sf);

    if vm.normed {
        for i in 0..ii { for j in 0..jj {
            if vm.counts[[i, j]] < min_count as f32 { continue; }
            let v = vm.in_vecs.slice(s![i, j, ..]);
            let sim = qvec.dot(&v);
            heap.push((i as u32, j as u32, sim));
            if heap.len() > top_k { heap.pop(); }
        }}
    } else {
        for i in 0..ii { for j in 0..jj {
            if vm.counts[[i, j]] < min_count as f32 { continue; }
            let v = vm.in_vecs.slice(s![i, j, ..]);
            let sim = qvec.dot(&v) / norm_l2(&v);
            heap.push((i as u32, j as u32, sim));
            if heap.len() > top_k { heap.pop(); }
        }}
    }

    let mut hv: Vec<_> = heap.drain().collect();
    hv.sort_by(sf);
    hv
}

fn norm_l2v<D>(a: ArrayView<'_, f32, D>) -> f32
    where D: ndarray::Dimension,
{
    a.fold(0.0, |l, r| l + *r**r).sqrt()
}

pub fn nearest_mmul(vm: &VectorModel, head_id: usize, top_k: usize, min_count: usize, min_prob: f64) -> Vec<(usize, Vec<(u32, u32, f32)>)>
{
    let ii = vm.in_vecs.len_of(Axis(0));
    let jj = vm.in_vecs.len_of(Axis(1));

    let sf = |(_id1, _s1, sim1): &(u32, u32, f32), (_id2, _s2, sim2): &(u32, u32, f32)| {
        sim2.partial_cmp(sim1).unwrap_or_else(
            || match (sim2.is_nan(), sim1.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (false, true) => std::cmp::Ordering::Greater,
                (true, false) => std::cmp::Ordering::Less,
                (false, false) => panic!(),
            }
        )
    };

    let mut pi = Array::<f64, Ix1>::zeros(jj);
    let _n_senses = expected_pi(&vm.counts, vm.alpha, head_id as u32, &mut pi, min_prob);
    let senses: Vec<_> = pi.iter().enumerate().filter_map(|(sense, prob)| if *prob >= min_prob { Some(sense) } else { None }).collect();
    if senses.is_empty() {
        return vec![];
    }

    let qvecs_r = vm.in_vecs.slice(s![head_id, .., ..]).select(Axis(0), &senses);
    let qnorms = qvecs_r.map_axis(Axis(1), norm_l2v);
    let qvecs = qvecs_r / qnorms.insert_axis(Axis(1));

    let mut heaps: Vec<_> = (0..senses.len()).map(|_i| BinaryHeap::new_by(sf)).collect();

    if vm.normed {
        for i in 0..ii { for j in 0..jj {
            if vm.counts[[i, j]] < min_count as f32 { continue; }
            let v = vm.in_vecs.slice(s![i, j, ..]);
            let sim = qvecs.dot(&v);
            for qj in 0..senses.len() {
                heaps[qj].push((i as u32, j as u32, sim[qj]));
                if heaps[qj].len() > top_k { heaps[qj].pop(); }
            }
        }}
    } else {
        for i in 0..ii { for j in 0..jj {
            if vm.counts[[i, j]] < min_count as f32 { continue; }
            let v = vm.in_vecs.slice(s![i, j, ..]);
            let sim = qvecs.dot(&v) / norm_l2(&v);
            for qj in 0..senses.len() {
                heaps[qj].push((i as u32, j as u32, sim[qj]));
                if heaps[qj].len() > top_k { heaps[qj].pop(); }
            }
        }}
    }

    let hvs: Vec<_> = senses.into_iter().zip(heaps.iter_mut()).map(|(sense, heap)| {
            let mut hv: Vec<_> = heap.drain().collect();
            hv.sort_by(sf);
            (sense, hv)
        })
        .collect();
    hvs
}

pub fn build_dense_sense_pool(vm: &VectorModel, min_count: usize) -> DenseSensePool
{
    let ii = vm.in_vecs.len_of(Axis(0));
    let jj = vm.in_vecs.len_of(Axis(1));
    let dim = vm.in_vecs.len_of(Axis(2));

    let mut ids = Vec::<(u32, u32)>::new();
    let mut vecs = Vec::<f32>::new();

    for i in 0..ii { for j in 0..jj {
        if vm.counts[[i, j]] < min_count as f32 { continue; }
        ids.push((i as u32, j as u32));
        let v = vm.in_vecs.slice(s![i, j, ..]);
        vecs.extend(v.iter().copied());
    }}

    DenseSensePool { dim, ids, vecs }
}

fn push_top_k(best: &mut Vec<(usize, f32)>, cand_ix: usize, sim: f32, top_k: usize)
{
    if sim.is_nan() { return; }
    if best.len() < top_k { best.push((cand_ix, sim)); return; }

    let mut min_pos = 0usize;
    let mut min_sim = best[0].1;
    for (pos, &(_ix, s)) in best.iter().enumerate().skip(1) {
        if s < min_sim {
            min_sim = s;
            min_pos = pos;
        }
    }
    if sim > min_sim {
        best[min_pos] = (cand_ix, sim);
    }
}

pub fn nearest_mmul_dense_pool(vm: &VectorModel, pool: &DenseSensePool, head_id: usize, top_k: usize, min_prob: f64) -> Vec<(usize, Vec<(u32, u32, f32)>)>
{
    debug_assert!(vm.normed);
    let jj = vm.in_vecs.len_of(Axis(1));

    let mut pi = Array::<f64, Ix1>::zeros(jj);
    let _n_senses = expected_pi(&vm.counts, vm.alpha, head_id as u32, &mut pi, min_prob);
    let senses: Vec<_> = pi.iter().enumerate().filter_map(|(sense, prob)| if *prob >= min_prob { Some(sense) } else { None }).collect();
    if senses.is_empty() || pool.is_empty() || top_k == 0 {
        return vec![];
    }

    let dim = pool.dim;
    debug_assert_eq!(dim, vm.in_vecs.len_of(Axis(2)));

    let qvecs = vm.in_vecs.slice(s![head_id, .., ..]).select(Axis(0), &senses);

    let mut bests: Vec<Vec<(usize, f32)>> = (0..senses.len()).map(|_i| Vec::with_capacity(top_k)).collect();

    let n_cands = pool.ids.len();
    let cand = ArrayView2::from_shape((n_cands, dim), pool.vecs.as_slice()).expect("dense pool vecs must be contiguous");

    const BLOCK: usize = 4096;
    let mut sims = Array2::<f32>::zeros((senses.len(), BLOCK));

    let mut start = 0usize;
    while start < n_cands {
        let end = (start + BLOCK).min(n_cands);
        let width = end - start;

        let cand_block = cand.slice(s![start..end, ..]);
        let mut sims_view = sims.slice_mut(s![.., 0..width]);
        sims_view.fill(0.0);
        ndarray::linalg::general_mat_mul(1.0, &qvecs, &cand_block.t(), 0.0, &mut sims_view);

        for qj in 0..senses.len() {
            for bi in 0..width {
                let cand_ix = start + bi;
                push_top_k(&mut bests[qj], cand_ix, sims_view[(qj, bi)], top_k);
            }
        }

        start = end;
    }

    let mut out = Vec::with_capacity(senses.len());
    for (qj, &sense) in senses.iter().enumerate() {
        let mut hv = bests[qj].iter().map(|&(cand_ix, sim)| {
            let (i, j) = pool.ids[cand_ix];
            (i, j, sim)
        }).collect::<Vec<_>>();
        hv.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        out.push((sense, hv));
    }
    out
}
