use wide::f32x8;

#[inline(always)]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut sum = f32x8::ZERO;
    let mut i = 0usize;

    while i + 8 <= a.len() {
        let av: [f32; 8] =
            unsafe { core::ptr::read_unaligned(a.as_ptr().add(i) as *const [f32; 8]) };
        let bv: [f32; 8] =
            unsafe { core::ptr::read_unaligned(b.as_ptr().add(i) as *const [f32; 8]) };
        sum = sum + f32x8::from(av) * f32x8::from(bv);
        i += 8;
    }

    let mut total = sum.reduce_add();
    for j in i..a.len() {
        total += a[j] * b[j];
    }
    total
}

#[inline(always)]
pub(crate) fn axpy_f32(y: &mut [f32], a: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());

    let av = f32x8::splat(a);
    let mut i = 0usize;

    while i + 8 <= y.len() {
        let yv: [f32; 8] =
            unsafe { core::ptr::read_unaligned(y.as_ptr().add(i) as *const [f32; 8]) };
        let xv: [f32; 8] =
            unsafe { core::ptr::read_unaligned(x.as_ptr().add(i) as *const [f32; 8]) };
        let r = f32x8::from(yv) + f32x8::from(xv) * av;
        unsafe { core::ptr::write_unaligned(y.as_mut_ptr().add(i) as *mut [f32; 8], r.to_array()) };
        i += 8;
    }

    for j in i..y.len() {
        y[j] += a * x[j];
    }
}
