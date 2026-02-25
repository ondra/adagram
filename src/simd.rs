use wide::AlignTo;
use wide::f32x8;

#[inline(always)]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // Safety: `wide::f32x8` is a plain-old-data SIMD wrapper and we only read initialized f32s.
    let (a_left, a_mid, a_right) = f32x8::simd_align_to(a);
    let (b_left, b_mid, b_right) = f32x8::simd_align_to(b);
    assert_eq!(a_left.len(), b_left.len(), "misaligned SIMD split (left)");
    assert_eq!(a_mid.len(), b_mid.len(), "misaligned SIMD split (mid)");
    assert_eq!(
        a_right.len(),
        b_right.len(),
        "misaligned SIMD split (right)"
    );

    let mut total = 0.0f32;
    for i in 0..a_left.len() {
        total += a_left[i] * b_left[i];
    }

    let mut sum = f32x8::ZERO;
    for (av, bv) in a_mid.iter().zip(b_mid.iter()) {
        sum = sum + (*av) * (*bv);
    }
    total += sum.reduce_add();

    for i in 0..a_right.len() {
        total += a_right[i] * b_right[i];
    }

    total
}

#[inline(always)]
pub(crate) fn axpy_f32(y: &mut [f32], a: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());

    let av = f32x8::splat(a);
    // Safety: same rationale as in `dot_f32` above; additionally the aligned middle is written
    // back as `f32x8` without violating aliasing (it is the same memory as `y`).
    let (y_left, y_mid, y_right) = f32x8::simd_align_to_mut(y);
    let (x_left, x_mid, x_right) = f32x8::simd_align_to(x);
    assert_eq!(y_left.len(), x_left.len(), "misaligned SIMD split (left)");
    assert_eq!(y_mid.len(), x_mid.len(), "misaligned SIMD split (mid)");
    assert_eq!(
        y_right.len(),
        x_right.len(),
        "misaligned SIMD split (right)"
    );

    for i in 0..y_left.len() {
        y_left[i] += a * x_left[i];
    }

    for (yv, xv) in y_mid.iter_mut().zip(x_mid.iter()) {
        *yv = *yv + (*xv) * av;
    }

    for i in 0..y_right.len() {
        y_right[i] += a * x_right[i];
    }
}
