use wide::f32x8;

#[inline(always)]
fn prefix_to_align_32(ptr: *const f32, len: usize) -> usize {
    const ALIGN: usize = 32;
    const ELEM: usize = core::mem::size_of::<f32>();
    debug_assert_eq!(ELEM, 4);

    let addr = ptr as usize;
    let mis = addr & (ALIGN - 1);
    if mis == 0 {
        0
    } else {
        let bytes = ALIGN - mis;
        let elems = bytes / ELEM;
        core::cmp::min(elems, len)
    }
}

#[inline(always)]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    const LANES: usize = 8;
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let prefix = prefix_to_align_32(a_ptr, len);
    let mut total = 0.0f32;
    for i in 0..prefix {
        total += a[i] * b[i];
    }

    let mut i = prefix;
    let end = len - ((len - i) % LANES);

    // If the two pointers have the same mod-32 offset, then aligning `a` also aligns `b`.
    let aligned_b = ((a_ptr as usize) ^ (b_ptr as usize)) & 31 == 0;

    let mut sum = f32x8::ZERO;
    unsafe {
        while i < end {
            let av = core::ptr::read(a_ptr.add(i) as *const f32x8);
            let bv = if aligned_b {
                core::ptr::read(b_ptr.add(i) as *const f32x8)
            } else {
                core::ptr::read_unaligned(b_ptr.add(i) as *const f32x8)
            };
            sum = sum + av * bv;
            i += LANES;
        }
    }

    total += sum.reduce_add();
    for j in i..len {
        total += a[j] * b[j];
    }
    total
}

#[inline(always)]
pub(crate) fn axpy_f32(y: &mut [f32], a: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());

    const LANES: usize = 8;
    let len = y.len();
    let y_ptr = y.as_mut_ptr();
    let x_ptr = x.as_ptr();

    let prefix = prefix_to_align_32(y_ptr, len);
    for i in 0..prefix {
        y[i] += a * x[i];
    }

    let mut i = prefix;
    let end = len - ((len - i) % LANES);

    let av = f32x8::splat(a);
    let aligned_x = ((y_ptr as usize) ^ (x_ptr as usize)) & 31 == 0;

    unsafe {
        while i < end {
            let yv = core::ptr::read(y_ptr.add(i) as *const f32x8);
            let xv = if aligned_x {
                core::ptr::read(x_ptr.add(i) as *const f32x8)
            } else {
                core::ptr::read_unaligned(x_ptr.add(i) as *const f32x8)
            };
            let r = yv + xv * av;
            core::ptr::write(y_ptr.add(i) as *mut f32x8, r);
            i += LANES;
        }
    }

    for j in i..len {
        y[j] += a * x[j];
    }
}
