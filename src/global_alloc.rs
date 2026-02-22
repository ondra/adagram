// Shared global allocator setup for all binaries in this package.
//
// Binaries include this file via:
//   #[path = "../global_alloc.rs"]
//   mod global_alloc;

#[cfg(all(feature = "jemalloc", feature = "mimalloc"))]
compile_error!("Features `jemalloc` and `mimalloc` are mutually exclusive; enable at most one.");

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
