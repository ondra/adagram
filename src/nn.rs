use crate::adagram::VectorModel;
use crate::common::expected_pi;
use crate::simd;
use binary_heap_plus::BinaryHeap;
use ndarray::prelude::*;
use multiversion::multiversion;

#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct AlignedChunk([f32; 8]);

pub struct DenseSensePool {
    dim: usize,
    ids: Vec<(u32, u32)>,
    vecs: Vec<AlignedChunk>,
}

impl DenseSensePool {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    fn vecs_f32(&self) -> &[f32] {
        if self.vecs.is_empty() {
            return &[];
        }
        unsafe {
            std::slice::from_raw_parts(self.vecs.as_ptr() as *const f32, self.vecs.len() * 8)
        }
    }
}

#[inline(always)]
fn norm_l2_slice(a: &[f32]) -> f32 {
    simd::dot_f32(a, a).sqrt()
}


#[multiversion(targets = "simd")]
/// make all input vectors unit length
pub fn vm_norm(vm: &mut VectorModel) {
    let ii = vm.in_vecs.len_of(Axis(0));
    let jj = vm.in_vecs.len_of(Axis(1));
    let dim = vm.in_vecs.len_of(Axis(2));
    let in_all = vm.in_vecs.as_slice_mut().unwrap();
    for i in 0..ii {
        for j in 0..jj {
            let offset = (i * jj + j) * dim;
            let v = &mut in_all[offset..offset + dim];
            let n = norm_l2_slice(v);
            v.iter_mut().for_each(|e| *e /= n);
        }
    }
    vm.normed = true;
}

#[multiversion(targets = "simd")]
pub fn sim(
    vm: &VectorModel,
    head_id1: usize,
    senseno1: usize,
    head_id2: usize,
    senseno2: usize,
) -> f32 {
    let in_all = vm.in_vecs.as_slice().unwrap();
    let jj = vm.in_vecs.len_of(Axis(1));
    let dim = vm.in_vecs.len_of(Axis(2));

    let off1 = (head_id1 * jj + senseno1) * dim;
    let s1 = &in_all[off1..off1 + dim];
    let off2 = (head_id2 * jj + senseno2) * dim;
    let s2 = &in_all[off2..off2 + dim];

    simd::dot_f32(s1, s2) / (norm_l2_slice(s1) * norm_l2_slice(s2))
}

#[multiversion(targets = "simd")]
pub fn sim_normed(
    vm: &VectorModel,
    head_id1: usize,
    senseno1: usize,
    head_id2: usize,
    senseno2: usize,
) -> f32 {
    let in_all = vm.in_vecs.as_slice().unwrap();
    let jj = vm.in_vecs.len_of(Axis(1));
    let dim = vm.in_vecs.len_of(Axis(2));

    let off1 = (head_id1 * jj + senseno1) * dim;
    let s1 = &in_all[off1..off1 + dim];
    let off2 = (head_id2 * jj + senseno2) * dim;
    let s2 = &in_all[off2..off2 + dim];

    simd::dot_f32(s1, s2)
}

#[multiversion(targets = "simd")]
pub fn nearest(
    vm: &VectorModel,
    head_id: usize,
    senseno: usize,
    top_k: usize,
    min_count: usize,
) -> Vec<(u32, u32, f32)> {
    let ii = vm.in_vecs.len_of(Axis(0));
    let jj = vm.in_vecs.len_of(Axis(1));
    let dim = vm.in_vecs.len_of(Axis(2));
    let in_all = vm.in_vecs.as_slice().unwrap();

    let q_off = (head_id * jj + senseno) * dim;
    let q = &in_all[q_off..q_off + dim];
    let qnorm = norm_l2_slice(q);

    let sf = |(_id1, _s1, sim1): &(u32, u32, f32), (_id2, _s2, sim2): &(u32, u32, f32)| {
        sim2.partial_cmp(sim1)
            .unwrap_or_else(|| match (sim2.is_nan(), sim1.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (false, true) => std::cmp::Ordering::Greater,
                (true, false) => std::cmp::Ordering::Less,
                (false, false) => panic!(),
            })
    };

    let mut heap = BinaryHeap::new_by(sf);

    if vm.normed {
        for i in 0..ii {
            for j in 0..jj {
                if vm.counts[[i, j]] < min_count as f32 {
                    continue;
                }
                let v_off = (i * jj + j) * dim;
                let v = &in_all[v_off..v_off + dim];
                let sim = simd::dot_f32(q, v) / qnorm;
                heap.push((i as u32, j as u32, sim));
                if heap.len() > top_k {
                    heap.pop();
                }
            }
        }
    } else {
        for i in 0..ii {
            for j in 0..jj {
                if vm.counts[[i, j]] < min_count as f32 {
                    continue;
                }
                let v_off = (i * jj + j) * dim;
                let v = &in_all[v_off..v_off + dim];
                let sim = simd::dot_f32(q, v) / (qnorm * norm_l2_slice(v));
                heap.push((i as u32, j as u32, sim));
                if heap.len() > top_k {
                    heap.pop();
                }
            }
        }
    }

    let mut hv: Vec<_> = heap.drain().collect();
    hv.sort_by(sf);
    hv
}

#[multiversion(targets = "simd")]
pub fn nearest_mmul(
    vm: &VectorModel,
    head_id: usize,
    top_k: usize,
    min_count: usize,
    min_prob: f64,
) -> Vec<(usize, Vec<(u32, u32, f32)>)> {
    let ii = vm.in_vecs.len_of(Axis(0));
    let jj = vm.in_vecs.len_of(Axis(1));
    let dim = vm.in_vecs.len_of(Axis(2));
    let in_all = vm.in_vecs.as_slice().unwrap();

    let sf = |(_id1, _s1, sim1): &(u32, u32, f32), (_id2, _s2, sim2): &(u32, u32, f32)| {
        sim2.partial_cmp(sim1)
            .unwrap_or_else(|| match (sim2.is_nan(), sim1.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (false, true) => std::cmp::Ordering::Greater,
                (true, false) => std::cmp::Ordering::Less,
                (false, false) => panic!(),
            })
    };

    let mut pi = Array::<f64, Ix1>::zeros(jj);
    let _n_senses = expected_pi(&vm.counts, vm.alpha, head_id as u32, &mut pi, min_prob);
    let senses: Vec<_> = pi
        .iter()
        .enumerate()
        .filter_map(|(sense, prob)| if *prob >= min_prob { Some(sense) } else { None })
        .collect();
    if senses.is_empty() {
        return vec![];
    }

    let q_slices: Vec<&[f32]> = senses
        .iter()
        .map(|&s| {
            let off = (head_id * jj + s) * dim;
            &in_all[off..off + dim]
        })
        .collect();
    let qnorms: Vec<f32> = q_slices.iter().map(|q| norm_l2_slice(q)).collect();

    let mut heaps: Vec<_> = (0..senses.len()).map(|_i| BinaryHeap::new_by(sf)).collect();

    if vm.normed {
        for i in 0..ii {
            for j in 0..jj {
                if vm.counts[[i, j]] < min_count as f32 {
                    continue;
                }
                let v_off = (i * jj + j) * dim;
                let v = &in_all[v_off..v_off + dim];
                for (qj, q) in q_slices.iter().enumerate() {
                    let sim = simd::dot_f32(q, v) / qnorms[qj];
                    heaps[qj].push((i as u32, j as u32, sim));
                    if heaps[qj].len() > top_k {
                        heaps[qj].pop();
                    }
                }
            }
        }
    } else {
        for i in 0..ii {
            for j in 0..jj {
                if vm.counts[[i, j]] < min_count as f32 {
                    continue;
                }
                let v_off = (i * jj + j) * dim;
                let v = &in_all[v_off..v_off + dim];
                let vnorm = norm_l2_slice(v);
                for (qj, q) in q_slices.iter().enumerate() {
                    let sim = simd::dot_f32(q, v) / (qnorms[qj] * vnorm);
                    heaps[qj].push((i as u32, j as u32, sim));
                    if heaps[qj].len() > top_k {
                        heaps[qj].pop();
                    }
                }
            }
        }
    }

    let hvs: Vec<_> = senses
        .into_iter()
        .zip(heaps.iter_mut())
        .map(|(sense, heap)| {
            let mut hv: Vec<_> = heap.drain().collect();
            hv.sort_by(sf);
            (sense, hv)
        })
        .collect();
    hvs
}

#[multiversion(targets = "simd")]
pub fn build_dense_sense_pool(vm: &VectorModel, min_count: usize) -> DenseSensePool {
    let ii = vm.in_vecs.len_of(Axis(0));
    let jj = vm.in_vecs.len_of(Axis(1));
    let dim = vm.in_vecs.len_of(Axis(2));
    let in_all = vm.in_vecs.as_slice().unwrap();
    let chunks_per_vec = dim / 8;
    debug_assert_eq!(dim % 8, 0);

    let mut ids = Vec::<(u32, u32)>::new();
    let mut vecs = Vec::<AlignedChunk>::new();

    for i in 0..ii {
        for j in 0..jj {
            if vm.counts[[i, j]] < min_count as f32 {
                continue;
            }
            ids.push((i as u32, j as u32));
            let off = (i * jj + j) * dim;
            let v = &in_all[off..off + dim];
            for c in 0..chunks_per_vec {
                let s = c * 8;
                let chunk: [f32; 8] = v[s..s + 8].try_into().unwrap();
                vecs.push(AlignedChunk(chunk));
            }
        }
    }

    DenseSensePool { dim, ids, vecs }
}

fn push_top_k(best: &mut Vec<(usize, f32)>, cand_ix: usize, sim: f32, top_k: usize) {
    if sim.is_nan() {
        return;
    }
    if best.len() < top_k {
        best.push((cand_ix, sim));
        return;
    }

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

#[multiversion(targets = "simd")]
pub fn nearest_mmul_dense_pool(
    vm: &VectorModel,
    pool: &DenseSensePool,
    head_id: usize,
    top_k: usize,
    min_prob: f64,
) -> Vec<(usize, Vec<(u32, u32, f32)>)> {
    debug_assert!(vm.normed);
    let jj = vm.in_vecs.len_of(Axis(1));
    let dim = pool.dim;
    debug_assert_eq!(dim, vm.in_vecs.len_of(Axis(2)));

    let mut pi = Array::<f64, Ix1>::zeros(jj);
    let _n_senses = expected_pi(&vm.counts, vm.alpha, head_id as u32, &mut pi, min_prob);
    let senses: Vec<_> = pi
        .iter()
        .enumerate()
        .filter_map(|(sense, prob)| if *prob >= min_prob { Some(sense) } else { None })
        .collect();
    if senses.is_empty() || pool.is_empty() || top_k == 0 {
        return vec![];
    }

    let in_all = vm.in_vecs.as_slice().unwrap();
    let pool_f32 = pool.vecs_f32();

    let q_slices: Vec<&[f32]> = senses
        .iter()
        .map(|&s| {
            let off = (head_id * jj + s) * dim;
            &in_all[off..off + dim]
        })
        .collect();

    let mut bests: Vec<Vec<(usize, f32)>> = (0..senses.len())
        .map(|_i| Vec::with_capacity(top_k))
        .collect();

    let n_cands = pool.ids.len();

    for ci in 0..n_cands {
        let cand = &pool_f32[ci * dim..(ci + 1) * dim];
        for (qj, q) in q_slices.iter().enumerate() {
            let sim = simd::dot_f32(q, cand);
            push_top_k(&mut bests[qj], ci, sim, top_k);
        }
    }

    let mut out = Vec::with_capacity(senses.len());
    for (qj, &sense) in senses.iter().enumerate() {
        let mut hv = bests[qj]
            .iter()
            .map(|&(cand_ix, sim)| {
                let (i, j) = pool.ids[cand_ix];
                (i, j, sim)
            })
            .collect::<Vec<_>>();
        hv.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        out.push((sense, hv));
    }
    out
}
