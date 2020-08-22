use criterion::{black_box, criterion_group, criterion_main, Criterion};

use llmalloc::LLAllocator;

static LL_ALLOCATOR: LLAllocator = LLAllocator::new();

//  Single-Threaded Single-Allocation Round-Trip.
//
//  This benchmarks repeatedly allocates and deallocates a block of memory on a single thread.
//
//  This is somewhat of a best-case scenario for thread-local caching and measures the lower-bound of allocator latency.
fn single_threaded_single_allocation_round_trip(c: &mut Criterion) {
    LL_ALLOCATOR.warm_up().expect("Warmed up");

    c.bench_function("ST SA RT - sys", |b| b.iter(|| {
        let _ = black_box(Vec::<u8>::with_capacity(32));
    }));
    c.bench_function("ST SA RT - ll", |b| {
        let layout = std::alloc::Layout::from_size_align(32, 1).expect("Valid Layout");
        b.iter(|| {
            let pointer = LL_ALLOCATOR.allocate(layout);
            let pointer = black_box(pointer);
            unsafe { LL_ALLOCATOR.deallocate(pointer) };
        });
    });
}

criterion_group!(benches, single_threaded_single_allocation_round_trip);
criterion_main!(benches);

//  FIXME: use sys crates... properly configured for system libraries.
#[link(name = "c")]
extern "C" {}

#[link(name = "numa")]
extern "C" {}

#[link(name = "pthread")]
extern "C" {}
