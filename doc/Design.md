#   Design

This document presents some of the design decisions that went into llmalloc, and why.

##  Why would anyone allocate, or deallocate, during a latency-sensitive operation?

Truth to be told? Ideally, they should _not_.

_In theory_, careful use of appropriately sized object pools, and other such strategies, can allow entire application
to completely eschew memory allocations.

_In practice_, such strategies are unwieldy, and not as well supported by languages and debugging tools, which makes
them _costly_ in terms of developer time.

In contrast, a low-latency memory allocator works _much better_:

-   Careful design allows avoiding allocation/deallocation in _most_ cases, so its raw performance is not as critical,
    though see the following point on _what low-latency means_.
-   Allocation/deallocation is well known, and well supported: standard smart pointers, standard containers, etc...
-   Allocation/deallocation is well supported.

The latter point is critical, a compile-time switch to disable the low-latency memory allocator immediately enables the
use of tools such as the various _saniziters_, or _valgrind_, to debug the various memory bugs -- as long as the
low-latency allocator itself is bug free, of course.


##  What does low-latency means?

Low-latency is often misconstrued. Obviously, low-latency means focusing on latency metrics, and not throughput metrics,
but which metrics?

In my line of work, the median, modes, or average are of course of interest, but the focus is on _tail_ latencies.
Indeed, often times, it is considered a good trade-off to see the median rise up by 10%, if it means improving the 90th,
95th, or 99th percentiles. Ideally, having a perfectly constant latency would be the bees knees.

The reason is that tail latencies compound: if there are 2 components A and B, with A having a 90th percentile latency
of 500ns, and B having a 90th percentile latency of 500ns, then in over 1% of cases the compound latency will be above
1us.

A low-latency application developer will therefore focus on the _tail_ cases to decide whether a particular operation
is suitable, or not, in the hot path.


##  Hardware

### Non-Uniform Memory Architectures

RAM stands for Random-Access Memory. Traditionally, Random-Access meant that access time was constant.

Nowadays, memory accesses are _far_ from constant time, for multiple reasons:

-   RAM accesses have not actually been constant time for a long time.
-   CPU cache and Out Of Order execution may or may not compensate.
-   Cache Coherency Protocols may or may not result in inter-core traffic, based on cache state.

NUMA further compounds the above issues:

-   There is a latency penalty in accessing another socket's or core's memory bank.
-   There is a latency penalty in coordinating cache coherency between sockets, compared to only between cores of a
    single socket.

llmalloc has been design with Mechanical Sympathy in mind:

-   _Sockets_ are designed to map to NUMA nodes, guaranteeing that memory allocations are drawn from the current NUMA
    node memory banks.
-   Inter-_Socket_ communications are limited; and notably deallocations are buffered locally and returned in batches.


### Cache misses

A cache miss is a latency penalty. llmalloc has no direct control over cache misses, but can still be cache friendly.

Being friendly to the instruction cache:

-   Avoid bloat: llmalloc's code contains a single instance of the generic llmalloc-core structures and functions,
    avoiding monomorphization bloat.
-   Avoid eviction: llmalloc vies to clearly separate hot code from cold code, to avoid dragging in seldom used
    code into the instruction cache... and evict application code.

Being friendly to the data cache:

-   Isolate local and foreign data: llmalloc's data layout clearly separates data that is read-only, thread-local, and
    global in order to avoid needless chatter between cores (and sockets).
-   Batch foreign accesses: in a producer-consumer scenario, memory is inevitably deallocated on a different thread, and
    possibly socket, than it was allocated. llmalloc batches the return of deallocated memory. This does not affect
    worst case latency, but does improve throughput.

Being friendly to the Translation Look-aside Buffer (TLB):

-   Huge Pages: llmalloc's core organization mirrors Linux' concept of Huge Pages and Large Pages to play to the TLB
    strengths and minimize TLB misses. On a suitably configured server, all memory allocated by llmalloc will be located
    in a handful of Huge Pages which will permanently reside in the TLB.


### Atomic Read-Modify-Write

Atomic operations are not a silver bullet:

-   Atomic operations _still_ require obtaining read/write access to the cache line, which may involve Cache Coherency
    communication across cores and sockets.
-   Atomic Read-Modify-Write operations tend to be expensive _even_ in the absence of contention.

Careful design will attempt to reduce the number of atomic operations, and in particular to avoid Read-Modify-Write
operations unless strictly necessary. Guarding a Read-Modify-Write operation by a simply load and a check can be worth
it.


##  Operating System

### System calls

System calls are the anti-thesis of low-latency calls. _Some_ system calls are accelerated on Linux with VDSO, `mmap`
and `munmap` are not of them.

Kernel calls are expensive and unbounded, beyond the switch to (and fro) kernel mode, the kernel may also decide to
perform arbitrary operations. For example, a request to allocate more memory may trigger the OOM killer.

As a result, low-latency applications will prefer to perform kernel calls only during the _startup_ and _shutdown_
sequences, where latency does not matter yet, and _no other call_ during normal operations.

llmalloc is design with such requirements in mind, allowing the application to pre-allocate Huge Pages ahead of time.

This decision has a functional consequence: unlike traditional malloc libraries, memory is _never_ relinquished to the
Operating System during normal operations.


##  Library

### Adrift

Casting pages adrift is llmalloc perhaps surprising solution to handling pages that are too full, and therefore must
wait for memory blocks to be freed before being able to handle allocations again.

The text-book solution would be to create a linked-list of such pages. When a page is too full, prepend or append it to
the list, then periodically walk the list searching for "empty enough" pages that can be used again for allocation. The
text-book solution is rather unsatisfactory as it involves an O(N) walk and contention between unrelated threads.

Solving the O(N) walk is simple. Whenever a memory-block is freed within the page, the thread which frees the block can
assess whether the page is now "empty enough" and if so remove it from the list. More interesting, though, is the new
question this brings forth: if there is no reason to ever walk the list, why is there a list in the first place?

The answer is that the list is _pointless_!

Instead, llmalloc opts for casting pages adrift when they are too full, and catching them when they are ready to handle
allocations again:

-   Each page contains an atomic boolean, indicating whether the page is adrift or not.
-   Upon exhausting the page, the allocating thread toggles the boolean and forgets about the page.
-   Any thread which returns memory blocks to the page checks if the page is adrift, and if so and enough memory blocks
    have been returned, catches the page and inserts it into the list of pages that are ready to handle allocations.

There are some race conditions to solve, both between casting the page adrift and catching it, and between potential
catchers, but the algorithm is still simple, and the principle is dead simple.

And it solves _both_ problems compared to the text-book solution:

-   There is no O(N) behavior, both casting and catching are O(1).
-   Contention is localized, only threads actively allocating and deallocating on this very page are potentially
    contending. Multiple threads casting multiple pages adrift do not interact in any way.

Only one source of contention is left: inserting the page back into the list of pages ready to handle allocations. It
can be limited by maintaining multiple lists -- one for each class size. It is unclear whether it can be entirely
eliminated; it may not be worth trying harder though.


### Batches

In general, a simple way to improve throughput, at the cost of increasing the variance of latency, is to batch
operations. Batches tend to play to a modern CPU strengths, and notably tend to be cache friendly.

llmalloc is a low-latency memory allocator, and is therefore primarily concerned about _latency_ rather than throughput.
As a result, llmalloc does _not_ use "amortized O(1)" algorithms.

llmalloc only uses _one_ batch operation, to reduce contention on deallocation. Deallocations are locally buffered,
and returned in a batch.

This particular operation, however, consists in pre-pending a singly-linked list to another singly-linked list, and
careful care has been taken for this operation to be O(1) with regard to the number of elements in either linked list
by caching the tail pointer of the list to be pre-pended.


### Contention

llmalloc leans heavily on thread-local data, and a thread-local cache of memory, to avoid contention.

_In theory_, doing so does not improve tail latencies, as it is always possible that _all_ cores could come to contend
at the same time.

_In practice_, sufficiently large thread-local caches do wonders to reduce the number of cores which actually come to
contend at any given time, and the more "random" the allocation and deallocation patterns of the application are, the
less contention is observed.


### Deallocation

malloc implementations tend to favor the performance of memory allocation at the expense of memory deallocation.
Expensive operations, such as consolidation of memory, is typically performed only during deallocation, leading to
a much higher latency variance for deallocation than for allocation.

llmalloc, instead, strives to treat both allocation and deallocation equally. Indeed, the very design of llmalloc has
been geared towards efficient deallocation: by making use of pointer alignment to efficiently process them.


### Lazy Initialization

Normal allocations in llmalloc are catered to by Large Pages. A single Large Page is initialized with a class size,
which determines the size of the memory blocks it will return, and the page will maintain a _stack_ of intrusively
linked memory blocks.

On x64/linux, a Large Page is 2MB and the smallest class size is 32 bytes. Given a header of about 1KB, this
translates into 65,504 memory blocks; which means that building the stack from scratch, even at an astounding 1ns per
element, would require about 65 _micro_ seconds. 65 times [an eternity](https://www.youtube.com/watch?v=NH1Tta7purM).

There are 2 solutions to the problem:

-   Preparing the Large Pages in advance.
-   Not building the entire stack at once.

Preparing the Large Pages in advance would be ideal, I suppose. My experience has been that it is impractical.

Instead, llmalloc uses a _watermark_ to indicate up to which points memory blocks have been allocated:

-   Memory before the watermark is managed by the stack.
-   Memory after the watermark has never been touched.

Then, anytime the stack is exhausted, it checks whether the watermark has reached the end of the page, and if not it
carves one memory block out of the raw memory, and bumps the watermark for next time.


### Wait Freedom

In general, lock freedom and wait freedom are not strictly necessary to guarantee low-latency in the context of
llmalloc: no application code is executed within the library itself.

There is one wrinkle there: the operating system is still free to interrupt an application thread at any point.
Disabling interrupts is one possibility, but it has its own cost.

llmalloc uses wait free algorithms instead, guaranteeing that even if _all but one_ threads are interrupted allocations
and deallocations on the non-interrupted thread can still proceed. At least as long as allocating, or deallocating, a
Huge Page is not necessary.
