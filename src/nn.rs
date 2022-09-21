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

pub fn nearest(vm: &VectorModel, head_id: usize, senseno: usize, top_k: usize, min_count: usize) -> Vec<(u32, u32, f32)>
{
    let ii = vm.in_vecs.len_of(Axis(0));
    let jj = vm.in_vecs.len_of(Axis(1));

    let qvec_r = vm.in_vecs.slice(s![head_id as usize, senseno, ..]);
    let qnorm = norm_l2(&qvec_r);
    let qvec = qvec_r.mapv(|v| v / qnorm) ;

    let sf = |(_id1, _s1, sim1): &(u32, u32, f32), (_id2, _s2, sim2): &(u32, u32, f32)| {
        sim2.partial_cmp(&sim1).unwrap_or_else(
            || match (sim2.is_nan(), sim1.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (false, true) => std::cmp::Ordering::Greater,
                (true, false) => std::cmp::Ordering::Less,
                (false, false) => panic!(),
            }
        )
    };

    let mut heap = BinaryHeap::new_by(sf);

    for i in 0..ii { for j in 0..jj {
        if vm.counts[[i, j]] < min_count as f32 { continue; }
        let v = vm.in_vecs.slice(s![i, j, ..]);
        let sim = qvec.dot(&v) / norm_l2(&v);
        heap.push((i as u32, j as u32, sim)); 
        if heap.len() > top_k { heap.pop(); }
    }}

    let mut hv: Vec<_> = heap.drain().collect();
    hv.sort_by(sf);
    hv
}
