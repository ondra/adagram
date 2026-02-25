use wide::AlignTo;

type SimdF32 = wide::f32x8;
const LANES: usize = core::mem::size_of::<SimdF32>() / core::mem::size_of::<<SimdF32 as AlignTo>::Elem>();

pub const PAD_DIM_TO_SIMD: bool = true;

/// If enabled, assumes all embedding slices are aligned to `SimdF32` and have a length that is a
/// multiple of `LANES`, so we can skip `simd_align_to{,_mut}` and any scalar edges.
pub const SIMD_ASSUME_ALIGNED_SLICES: bool = true;

#[inline(always)]
pub(crate) fn pad_dim(dim: usize) -> usize {
    if !PAD_DIM_TO_SIMD {
        return dim;
    }
    let rem = dim % LANES;
    if rem == 0 { dim } else { dim + (LANES - rem) }
}

#[inline(always)]
pub(crate) fn assert_simd_preconditions(dim_padded: usize, base_ptr: *const f32, what: &str) {
    if !SIMD_ASSUME_ALIGNED_SLICES {
        return;
    }
    assert_eq!(
        dim_padded % LANES,
        0,
        "{what}: dim_padded={dim_padded} not multiple of LANES={LANES}"
    );
    let align = core::mem::align_of::<SimdF32>();
    assert_eq!(
        (base_ptr as usize) & (align - 1),
        0,
        "{what}: base pointer not aligned to {align} bytes"
    );
}

#[inline(always)]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    if SIMD_ASSUME_ALIGNED_SLICES {
        debug_assert_eq!(a.len() % LANES, 0);
        debug_assert_eq!(b.len() % LANES, 0);
        debug_assert_eq!(
            (a.as_ptr() as usize) & (core::mem::align_of::<SimdF32>() - 1),
            0
        );
        debug_assert_eq!(
            (b.as_ptr() as usize) & (core::mem::align_of::<SimdF32>() - 1),
            0
        );

        let n = a.len() / LANES;
        let mut sum = SimdF32::ZERO;
        unsafe {
            let ap = a.as_ptr() as *const SimdF32;
            let bp = b.as_ptr() as *const SimdF32;
            for i in 0..n {
                let av = core::ptr::read(ap.add(i));
                let bv = core::ptr::read(bp.add(i));
                sum = sum + av * bv;
            }
        }
        return sum.reduce_add();
    }

    let (a_left, a_mid, a_right) = SimdF32::simd_align_to(a);
    let (b_left, b_mid, b_right) = SimdF32::simd_align_to(b);
    if a_left.len() != b_left.len() || a_mid.len() != b_mid.len() || a_right.len() != b_right.len()
    {
        return a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    }

    let mut total = 0.0f32;
    for i in 0..a_left.len() {
        total += a_left[i] * b_left[i];
    }

    let mut sum = SimdF32::ZERO;
    for i in 0..a_mid.len() {
        sum = sum + a_mid[i] * b_mid[i];
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

    if SIMD_ASSUME_ALIGNED_SLICES {
        let av = SimdF32::splat(a);
        debug_assert_eq!(y.len() % LANES, 0);
        debug_assert_eq!(x.len() % LANES, 0);
        debug_assert_eq!(
            (y.as_ptr() as usize) & (core::mem::align_of::<SimdF32>() - 1),
            0
        );
        debug_assert_eq!(
            (x.as_ptr() as usize) & (core::mem::align_of::<SimdF32>() - 1),
            0
        );

        let n = y.len() / LANES;
        unsafe {
            let yp = y.as_mut_ptr() as *mut SimdF32;
            let xp = x.as_ptr() as *const SimdF32;
            for i in 0..n {
                let yv = core::ptr::read(yp.add(i));
                let xv = core::ptr::read(xp.add(i));
                core::ptr::write(yp.add(i), yv + xv * av);
            }
        }
        return;
    }

    let (y_left, y_mid, y_right) = SimdF32::simd_align_to_mut(y);
    let (x_left, x_mid, x_right) = SimdF32::simd_align_to(x);
    if y_left.len() != x_left.len() || y_mid.len() != x_mid.len() || y_right.len() != x_right.len()
    {
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += a * xi;
        }
        return;
    }

    for i in 0..y_left.len() {
        y_left[i] += a * x_left[i];
    }

    let av = SimdF32::splat(a);
    for i in 0..y_mid.len() {
        y_mid[i] = y_mid[i] + x_mid[i] * av;
    }

    for i in 0..y_right.len() {
        y_right[i] += a * x_right[i];
    }
}
