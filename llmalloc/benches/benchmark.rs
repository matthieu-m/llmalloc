use std::{
    alloc::Layout,
    cmp,
    collections::VecDeque,
    ptr::NonNull,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};

use llmalloc::LLAllocator;
use llmalloc_test::BurstyBuilder;

static LL_ALLOCATOR: LLAllocator = LLAllocator::new();

//  Single-Thread Single-Allocation
//
//  This benchmark repeatedly allocates a block of memory on a single thread.
//
//  This is somewhat of a best-case scenario for thread-local caching and measures the lower-bound of allocator latency.
fn single_threaded_single_allocation_allocation(c: &mut Criterion) {
    fn bencher<T: Vector>(name: &str, c: &mut Criterion) {
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
    //  For reasons unfathomable, this benchmark is _completely_ wrecked on my VM, but works fine on regular hosts.
    //
    //  It consistently returns a measurement of about 5 _micro_ seconds, when allocate+deallocate is under 100ns.
    //
    //  Attempts:
    //  -   Valgrind clean, so no obvious memory error.
    //  -   Measurements using lfence + rdtsc (cycles, instead of nanos), lead to the exact same issue _and
    //      measurements_.
    //      -   The assembly only shows an (indirect) call to `LLAllocator::deallocate` within the timed section.
    //  -   Printing 1% and 99% percentiles show a range from 5us to 13us !?!?
    fn bencher<T: Vector>(name: &str, c: &mut Criterion) {
        c.bench_function(name, |b| b.iter_custom(|iterations| {
            let mut duration = Duration::default();

            for _ in 0..iterations {
                let v = black_box(T::with_capacity(32));

                let start = Instant::now();

                std::mem::drop(v);

                duration += start.elapsed();
            }

            duration
        }));
    }

    LL_ALLOCATOR.warm_up().expect("Warmed up");

    bencher::<SysVec>("ST SA Deallocation - sys", c);

    bencher::<LLVec>("ST SA Deallocation - ll", c);
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
    fn bencher<T: Vector>(name: &str, c: &mut Criterion, number_iterations: usize) {
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
    fn bencher<T: Vector>(name: &str, c: &mut Criterion, number_iterations: usize) {
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
    fn bencher<T: Vector>(name: &str, c: &mut Criterion, number_iterations: usize) {
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

//  Multi-Threaded Batch-Allocation Allocation.
//
//  This benchmark allocates N blocks of memory _per producer_ on N threads in parallel.
fn multi_threaded_batch_allocation_allocation(c: &mut Criterion) {
    fn bencher<T: Vector + Send + 'static>(name: &str, producers: usize, c: &mut Criterion) {
        bench_function_worst_of_parallel(
            name,
            producers,
            c,
            |iterations| (0..iterations).map(|_| None).collect::<Vec<Option<T>>>(),
            || |mut vec| {
                vec.iter_mut().for_each(|t| *t = Some(black_box(T::with_capacity(32))));
                vec
            }
        );
    }

    LL_ALLOCATOR.warm_up().expect("Warmed up");

    let number_cpus = num_cpus::get();

    for i in 0.. {
        let number_producers = 1usize << i;

        if number_producers >= number_cpus {
            return;
        }

        bencher::<SysVec>(
            &format!("PC BA Allocation {}P - sys", number_producers),
            number_producers,
            c
        );

        bencher::<LLVec>(
            &format!("PC BA Allocation {}P - ll", number_producers),
            number_producers,
            c
        );
    }
}

//  Multi-Threaded Batch-Allocation Deallocation.
//
//  This benchmark allocates N blocks of memory _per consumer_ on a single thread, then sends them to N consumer
//  threads which deallocates them in parallel.
fn multi_threaded_batch_allocation_deallocation(c: &mut Criterion) {
    fn bencher<T: Vector + Send + 'static>(name: &str, consumers: usize, c: &mut Criterion) {
        bench_function_worst_of_parallel(
            name,
            consumers,
            c,
            |iterations| (0..iterations)
                .map(|_| black_box(T::with_capacity(32)))
                .collect::<Vec<_>>(),
            || |vec| std::mem::drop(vec)
        );
    }

    LL_ALLOCATOR.warm_up().expect("Warmed up");

    let number_cpus = num_cpus::get();

    for i in 0.. {
        let number_consumers = 1usize << i;

        if number_consumers >= number_cpus {
            return;
        }

        bencher::<SysVec>(
            &format!("PC BA Deallocation {}C - sys", number_consumers),
            number_consumers,
            c
        );

        bencher::<LLVec>(
            &format!("PC BA Deallocation {}C - ll", number_consumers),
            number_consumers,
            c
        );
    }
}

criterion_group!(
    multi_threaded_batch_allocation,
    multi_threaded_batch_allocation_allocation,
    multi_threaded_batch_allocation_deallocation
);

criterion_main!(
    single_threaded_single_allocation,
    single_threaded_batch_allocation,
    multi_threaded_batch_allocation
);

//
//  Benchmark Helpers
//

//  This helper benchmarks the performance of `number_thread` instances of `G` running in parallel, repeatedly.
//
//  Under the hood, this function uses `Criterion::bench_function` and `Bencher::iter_custom`. For each iteration:
//  -   It calls `setup` to generate 1 `input` per thread, passing the number of iterations indicated by Criterion.
//  -   It calls `factory` to generate 1 `victim` per thread.
//  -   It spawns a thread in which it calls the `victim` with the `input`, measuring the elapsed time.
//
//  In each iteration, the threads are coordinated by a `rendez_vous` variable; they will busy-wait until the signal
//  and only then run, maximizing concurrency -- and thus contention.
//
//  The final measurement of each call to `Bencher::iter_custom` is the maximum duration experienced by any of the
//  threads.
//
//  Note:   The `victim` may feel free to return a value; the time taken to drop the value is _not_ part of the
//          measurement.
fn bench_function_worst_of_parallel<F, S, T, V, Z>(
    name: &str,
    number_threads: usize,
    c: &mut Criterion,
    mut setup: S,
    mut factory: F,
)
    where
        F: FnMut() -> V,
        S: FnMut(usize) -> T,
        T: Send + 'static,
        V: FnOnce(T) -> Z + Send + 'static,
{
    assert!(number_threads >= 1);

    c.bench_function(name, move |b| b.iter_custom(|iterations| {
        let measurements: Vec<_> = (0..number_threads)
            .map(|_| AtomicU64::new(0))
            .collect();

        let locals: Vec<_> = (0..number_threads).collect();

        let mut builder = BurstyBuilder::new(measurements, locals);

        builder.add_complex_step(|| {
            let mut input = Some(setup(iterations as usize));
            let mut victim = Some(factory());

            let prep = move |_: &Vec<AtomicU64>, _: &mut usize| -> (T, V) {
                LL_ALLOCATOR.warm_up().expect("Warmed up");

                (input.take().unwrap(), victim.take().unwrap())
            };

            let step = |measurements: &Vec<AtomicU64>, index: &mut usize, (input, victim): (T, V)| {
                let start = Instant::now();

                let _large_drop = black_box(victim(black_box(input)));

                let duration = start.elapsed();

                measurements[*index].store(duration.as_nanos() as u64, Ordering::Relaxed);
            };

            (prep, step)
        });

        let bursty = builder.launch(1);

        bursty.join();

        let duration = bursty.global().into_iter()
            .map(|measurement| measurement.load(Ordering::Relaxed))
            .map(|nanos| Duration::from_nanos(nanos))
            .max()
            .expect("At least one element");

        cmp::max(duration, Duration::from_nanos(1))
    }));
}

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
    pointer: NonNull<u8>,
    #[allow(dead_code)]
    len: usize,
    #[allow(dead_code)]
    cap: usize,
}

impl Vector for LLVec {
    fn with_capacity(capacity: usize) -> LLVec {
        let pointer = LL_ALLOCATOR.allocate(layout(capacity, 1)).unwrap();
        LLVec { pointer, len: 0, cap: capacity }
    }
}

impl Drop for LLVec {
    fn drop(&mut self) {
        unsafe { LL_ALLOCATOR.deallocate(self.pointer) }
    }
}

unsafe impl Send for LLVec {}

fn layout(size: usize, alignment: usize) -> Layout {
    Layout::from_size_align(size, alignment).expect("Valid Layout")
}

//  FIXME: use sys crates... properly configured for system libraries.
#[link(name = "numa")]
extern "C" {}
