use std::{alloc::Layout, collections::VecDeque, time};

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};

use llmalloc::LLAllocator;

static LL_ALLOCATOR: LLAllocator = LLAllocator::new();

//  Single-Thread Single-Allocation
//
//  This benchmark repeatedly allocates a block of memory on a single thread.
//
//  This is somewhat of a best-case scenario for thread-local caching and measures the lower-bound of allocator latency.
fn single_threaded_single_allocation_allocation(c: &mut Criterion) {
    fn bencher<T: Vector>(name: &'static str, c: &mut Criterion) {
        c.bench_function(name, |b| b.iter_with_large_drop(
            || black_box(T::with_capacity(32))
        ));
    }

    LL_ALLOCATOR.warm_up().expect("Warmed up");

    bencher::<SysVec>("ST SA Allocation - sys", c);

    bencher::<LLVec>("ST SA Allocation - ll", c);
}

//  Single-Thread Single-Allocation
//
//  This benchmark repeatedly deallocates a block of memory on a single thread.
//
//  This is somewhat of a best-case scenario for thread-local caching and measures the lower-bound of allocator latency.
fn single_threaded_single_allocation_deallocation(c: &mut Criterion) {
    //  For reasons unfathomable, this benchmark is _completely_ wrecked.
    //
    //  It consistently returns a measurement of about 5 _micro_ seconds, when allocate+deallocate is under 100ns.
    //
    //  Attempts:
    //  -   Valgrind clean, so no obvious memory error.
    //  -   Measurements using lfence + rdtsc (cycles, instead of nanos), lead to the exact same issue _and
    //      measurements_.
    //      -   The assembly only shows an (indirect) call to `LLAllocator::deallocate` within the timed section.
    //  -   Printing 1% and 99% percentiles show a range from 5us to 13us !?!?
    fn bencher<T: Vector>(name: &'static str, c: &mut Criterion) {
        c.bench_function(name, |b| b.iter_custom(|iterations| {
            let mut duration = time::Duration::default();

            for _ in 0..iterations {
                let v = black_box(T::with_capacity(32));

                let start = time::Instant::now();

                std::mem::drop(v);

                duration += start.elapsed();
            }

            duration
        }));
    }

    LL_ALLOCATOR.warm_up().expect("Warmed up");

    bencher::<SysVec>("ST SA Deallocation - sys (unexplained)", c);

    bencher::<LLVec>("ST SA Deallocation - ll (unexplained)", c);
}

//  Single-Threaded Single-Allocation Round-Trip.
//
//  This benchmark repeatedly allocates and deallocates a block of memory on a single thread.
//
//  This is somewhat of a best-case scenario for thread-local caching and measures the lower-bound of allocator latency.
fn single_threaded_single_allocation_round_trip(c: &mut Criterion) {
    LL_ALLOCATOR.warm_up().expect("Warmed up");

    c.bench_function("ST SA Round-trip - sys", |b| b.iter(|| {
        let _ = black_box(SysVec::with_capacity(32));
    }));
    c.bench_function("ST SA Round-trip - ll", |b| b.iter(|| {
        let _ = black_box(LLVec::with_capacity(32));
    }));
}

criterion_group!(
    single_threaded_single_allocation,
    single_threaded_single_allocation_allocation,
    single_threaded_single_allocation_deallocation,
    single_threaded_single_allocation_round_trip
);

//  Single-Thread Batch-Allocation Allocation.
//
//  This benchmark repeatedly allocates a block of memory on a single thread.
//
//  This is somewhat of a best-case scenario for thread-local caching and measures the lower-bound of allocator latency.
fn single_threaded_batch_allocation_allocation(c: &mut Criterion) {
    fn bencher<T: Vector>(name: &'static str, c: &mut Criterion, number_iterations: usize) {
        c.bench_function(name, |b| b.iter_batched_ref(
            || Vec::<T>::with_capacity(number_iterations),
            |v| v.push(black_box(T::with_capacity(32))),
            BatchSize::NumIterations(number_iterations as u64)
        ));
    }

    const NUMBER_ITERATIONS: usize = 1024;

    LL_ALLOCATOR.warm_up().expect("Warmed up");

    bencher::<SysVec>("ST BA Allocation - sys", c, NUMBER_ITERATIONS);

    bencher::<LLVec>("ST BA Allocation - ll", c, NUMBER_ITERATIONS);
}

//  Single-Thread Batch-Allocation Deallocation.
//
//  This benchmark repeatedly allocates a block of memory on a single thread.
//
//  This is somewhat of a best-case scenario for thread-local caching and measures the lower-bound of allocator latency.
fn single_threaded_batch_allocation_deallocation(c: &mut Criterion) {
    fn bencher<T: Vector>(name: &'static str, c: &mut Criterion, number_iterations: usize) {
        c.bench_function(name, |b| b.iter_batched_ref(
            || {
                let mut v = Vec::<T>::new();
                v.resize_with(number_iterations, || black_box(T::with_capacity(32)));
                v
            },
            |v| v.pop(),
            BatchSize::NumIterations(number_iterations as u64)
        ));
    }

    const NUMBER_ITERATIONS: usize = 1024;

    LL_ALLOCATOR.warm_up().expect("Warmed up");

    bencher::<SysVec>("ST BA Deallocation - sys", c, NUMBER_ITERATIONS);

    bencher::<LLVec>("ST BA Deallocation - ll", c, NUMBER_ITERATIONS);
}

//  Single-Thread Batch-Allocation Round-Trip.
//
//  This benchmark repeatedly allocates a block of memory on a single thread, then repeatedly deallocates them.
//
//  This is somewhat of a best-case scenario for thread-local caching and measures the lower-bound of allocator latency.
fn single_threaded_batch_allocation_round_trip(c: &mut Criterion) {
    fn bencher<T: Vector>(name: &'static str, c: &mut Criterion, number_iterations: usize) {
        c.bench_function(name, |b| b.iter_batched_ref(
            || {
                let mut v = VecDeque::<T>::with_capacity(number_iterations);
                v.resize_with(number_iterations - 1, || black_box(T::with_capacity(32)));
                v
            },
            |v| {
                v.push_back(black_box(T::with_capacity(32)));
                v.pop_front()
            },
            BatchSize::NumIterations(number_iterations as u64)
        ));
    }

    const NUMBER_ITERATIONS: usize = 1024;

    LL_ALLOCATOR.warm_up().expect("Warmed up");

    bencher::<SysVec>("ST BA Round-trip - sys", c, NUMBER_ITERATIONS);

    bencher::<LLVec>("ST BA Round-trip - ll", c, NUMBER_ITERATIONS);
}

criterion_group!(
    single_threaded_batch_allocation,
    single_threaded_batch_allocation_allocation,
    single_threaded_batch_allocation_deallocation,
    single_threaded_batch_allocation_round_trip
);

criterion_main!(
    single_threaded_single_allocation,
    single_threaded_batch_allocation
);

//
//  Implementation Details
//

trait Vector: Sized {
    fn with_capacity(capacity: usize) -> Self;
}

type SysVec = Vec<u8>;

impl Vector for SysVec {
    fn with_capacity(capacity: usize) -> SysVec { SysVec::with_capacity(capacity) }
}

//  Similar layout to Vec, for fairness.
struct LLVec {
    pointer: *mut u8,
    #[allow(dead_code)]
    len: usize,
    #[allow(dead_code)]
    cap: usize,
}

impl Vector for LLVec {
    fn with_capacity(capacity: usize) -> LLVec {
        let pointer = LL_ALLOCATOR.allocate(layout(capacity, 1));
        LLVec { pointer, len: 0, cap: capacity }
    }
}

impl Drop for LLVec {
    fn drop(&mut self) {
        unsafe { LL_ALLOCATOR.deallocate(self.pointer) }
    }
}

fn layout(size: usize, alignment: usize) -> Layout {
    Layout::from_size_align(size, alignment).expect("Valid Layout")
}

//  FIXME: use sys crates... properly configured for system libraries.
#[link(name = "numa")]
extern "C" {}
