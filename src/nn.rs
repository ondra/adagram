use binary_heap_plus::BinaryHeap;
use ndarray::prelude::*;
use crate::adagram::VectorModel; 

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
