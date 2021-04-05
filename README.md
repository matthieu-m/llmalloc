#   llmalloc - low-latency memory allocator

llmalloc is an alternative to malloc for low-latency environments.

If you are looking for a general-purpose replacement for malloc, prefer jemalloc or tcmalloc. The list of non-goals
should convince you to.

##  Goals

The primary goal of this alternative to malloc is low-latency memory allocation and deallocation.

Supplementary Goals:

-   Low cache footprint: llmalloc is a support library, it strives to minimize its cache footprint, both cache and data.
-   Wait-free allocation: llmalloc-core is itself wait-free, guaranteeing that even if all but one thread are blocked,
    the lone running thread can still allocate and deallocate on its own. The guarantee partially extends to llmalloc
    itself, with the exception of system calls.

Non-goals:

-   High throughput: llmalloc will always favor latency, as usual it comes at the expense of pure throughput.
-   Low-latency system calls: system calls are out of the purview of llmalloc, so instead the API provides ways to
    reserve memory ahead of time, so that no further system call is necessary until shutdown.
-   Memory efficiency: on x64/linux, llmalloc will reserve memory by increment of 1GB at a time, using Huge Pages if
    available.

Limitations:

-   Memory frugality: llmalloc cannot, by design, relinquish any allocated page of memory back to the OS until
    shutdown, it does not keep track of the necessary information.
-   Metrics: llmalloc does not provide any metric on actual memory usage, it does not keep track of such information.
-   Portability: llmalloc is only available on x64/linux platforms at the moment.

While the limitations could, potentially, be lifted, there is currently no intent to do so.

##  Structure of the repository

This repository contains 3 libraries:

-   llmalloc-core: the unopinionated core library, which contains the building bricks.
-   llmalloc: an opinionated implementation.
-   llmalloc-c: C bindings for llmalloc.

##  Maturity

llmalloc is in _alpha_ state; of note:

-   llmalloc-core:
    -   Audit: not audited.
    -   Fuzzing: fuzzed.
    -   Testing: good single-thread and multi-thread unit-test support.

-   llmalloc:
    -   Audit: not audited.
    -   Benchmarks: benchmarked, with performance on par or better than the system allocator.
    -   Fuzzing: fuzzed.
    -   Testing: one single-threaded integration test.

The library is ready to be trialed in realistic setups, but is not mature enough to drop in in a production application
without a qualification phase.

##  Design Sketch

A quick overview of the design of the library, to pique your interest.

For a more extensive discussion of design decisions, see [doc/Design.md](doc/Design.md).

### Concepts

llmalloc is organized around the concept of 3 allocation categories and 2 page sizes:

-   Large Pages: pages from which Normal allocations are served.
-   Huge Pages: pages from which Large allocations are served.

Huge Pages are provided by Huge allocations, and Large Pages by Large allocations.

### Key Pieces

At the root of llmalloc-core lies the _Domain_. Simply said, the _Domain_ defines the boundaries of memory, all
allocations performed within a _Domain_ can be deallocated only with the same _Domain_. The _Domain_ is host to an
instance of the `Platform` trait: a trait which defines how to allocate and deallocate Huge allocations, and therefore
Huge Pages.

A _Domain_ may have multiple _Sockets_. As the name implies, it is expected that each socket on the motherboard feature
one _Socket_. Non-Uniform Memory Architecture places a tax on inter-socket cache-coherence traffic, and therefore it
is best not to write to the same piece of memory from multiple sockets. Each _Socket_ will therefore keep track of its
own set of Huge Pages, from which all its allocations will come from, as well as provide a cache of Large Pages.

And finally, a _Socket_ can allocate multiple _Threads_. As the name implies, it is expected that each software thread
feature one _Thread_. Each _Thread_ will keep one Large Page for each class size of Normal allocations that it may
serve, in order to provide thread-local uncontended allocation and deallocation -- most of the time.

### Implementation of Deallocation

Unfortunately, the interface of the C function `free`, and the C++ operator `delete`, are not friendly: they hide
one critical piece of information, the size of the memory area to be returned to the memory allocator.

llmalloc borrows a trick from [mimalloc](https://github.com/microsoft/mimalloc), and encodes the category of the
allocation (Normal, Large, or Huge) in the _alignment_ of the pointer. This places some constraints on the allocation
of the Huge Pages, and enables low-latency deallocation.

On x64/linux, for example:

-   Normal allocations have an alignment strictly less than 2MB -- the alignment of Large Pages.
-   Large allocations have an alignment greater than or equal to 2MB, and strictly less than 1GB -- the alignment of
    Huge Pages.
-   Huge allocations have an alignment greater than or equal to 1GB.

And while the exact alignments may vary from platform to platform, the principles remain the same on all. As a result
finding the page to which an allocation belongs is as simple as masking its pointer lower bits by the appropriate mask,
which is deduced from the alignment.

##  Acknowledgements

The spark came from reading about the design of [mimalloc](https://github.com/microsoft/mimalloc), in particular two
key ideas resonated within me:

-   Alignment is information: the idea of encoding the category of an allocation (Normal, Large, or Huge) within the
    alignment of the pointer was an eye opener; it vastly simplifies unsized deallocation.
-   Cache footprint: the idea that a support library should vie for as low a cache footprint as possible, both an
    instruction cache footprint, and a data cache footprint, was also a light bulb moment. A key difference between
    micro-benchmarks and real-life applications is that in micro-benchmarks the library has all the cache to itself.
