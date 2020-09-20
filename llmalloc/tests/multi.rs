use std::{alloc::Layout, collections, ops, sync::{self, atomic}, thread};

use serial_test::serial;

use llmalloc::LLAllocator;

static LL_ALLOCATOR: LLAllocator = LLAllocator::new();

//
//  Tests
//

#[serial]
#[test]
fn acquire_release_thread_handles() {
    //  Test that acquiring/releasing thread handles from multiple threads concurrently works as expected.

    let number_iterations = number_iterations();
    let number_threads = number_threads();

    let mut thread_handles = collections::BTreeSet::new();

    for _ in 0..number_iterations {
        let start = RendezVous::new("start", number_threads);
        let end = RendezVous::new("end", number_threads);

        let pool = Pool::new(number_threads, |i| {
            let start = start.clone();
            let end = end.clone();

            move || {
                start.wait_until_all_ready();

                let pointer = Pointer::new(i);

                end.wait_until_all_ready();

                //  Sanity check, to ensure no other thread allocate the same pointer.
                assert_eq!(i, *pointer);

                LL_ALLOCATOR.thread_index()
            }
        });

        let results = pool.join();

        //  Ensure that no 2 threads obtained the same thread-local handle.
        let local_handles: collections::BTreeSet<_> = results.iter().cloned().collect();

        assert_eq!(number_threads, local_handles.len());

        //  Ensure that thread-local handles are reused across invocations:
        //  -   This ensures they are freed on thread destruction.
        //  -   This ensures they are reused, to avoid running out.
        if thread_handles.is_empty() {
            let _ = std::mem::replace(&mut thread_handles, local_handles);
            continue;
        }

        assert_eq!(thread_handles, local_handles);
    }
}

#[serial]
#[test]
fn producer_consumer_ring() {
    //  Test that blocks can be concurrenty allocated and deallocated, including deallocated on a separate thread.
    //
    //  The test is slightly convoluted, so a high-level overview is provided:
    //
    //  1.  "Victims" are prepared, those are numbers 0 to N made into `String`.
    //  2.  Concurrently (synchronized) those victims are moved into `Pointer`, each requiring an allocation.
    //  3.  The allocations are "shuffled", so that each vector contains pointers from alternating allocating threads.
    //  4.  Concurrently (synchronized) those shuffled pointers are freed, and their values recovered.
    //  5.  Check the recovered values match the originals, to ensure no corruption occurred.
    //  6.  The rendez_vous are reset, for the next iteration.

    fn create_victims(number: usize) -> Vec<String> { (0..number).map(|i| i.to_string()).collect() }

    fn push_victims(victims: Vec<String>, sink: &mut Vec<Pointer<String>>) {
        debug_assert!(sink.is_empty());

        for victim in victims {
            sink.push(Pointer::new(victim));
        }
    }

    fn pop_victims(stream: &mut Vec<Pointer<String>>, sink: &mut Vec<String>) {
        debug_assert!(stream.len() <= sink.capacity());

        stream.drain(..)
            .for_each(|mut pointer| sink.push(pointer.replace_with_default()));
    }

    #[inline(never)]
    fn shuffle_ring(ring: &[sync::Mutex<Vec<Pointer<String>>>]) {
        fn swap_head(vec: &mut Vec<&mut Pointer<String>>, index: usize) {
            debug_assert!(index > 0);

            let (head, tail) = vec.split_at_mut(index);
            let head: &mut Pointer<String> = head[0];
            let tail: &mut Pointer<String> = tail[0];

            std::mem::swap(head, tail);
        }

        let number_threads = ring.len();
        assert!(number_threads >= 2, "number_threads: {} < 2", number_threads);

        let number_victims = ring[0].try_lock().unwrap().len();

        let mut guards: Vec<_> = ring.iter().map(|mutex| mutex.try_lock().unwrap()).collect();

        for i in 0..number_victims {
            let shift = i % number_threads;

            if shift == 0 {
                continue;
            }

            let mut layer: Vec<&mut Pointer<String>> = guards.iter_mut().map(|guard| &mut guard[i]).collect();

            for _ in 0..shift {
                //  A single shift goes from [A, B, C, D] to [D, A, B, C].
                //  By iterating `shift` times, A shifts by that many places to the right.
                for target in 1..number_threads {
                    swap_head(&mut layer, target);
                }
            }
        }

    }

    let number_iterations = number_iterations();
    let number_threads = number_threads();
    let number_victims = 256;

    let allocation = RendezVous::new("allocation", number_threads);
    let deallocation = RendezVous::new("deallocation", number_threads);
    let shuffle_start = RendezVous::new("shuffle-start", number_threads);
    let shuffle_end = RendezVous::new("shuffle-end", number_threads);
    let next = RendezVous::new("next", 0);

    let ring = {
        let mut ring = Vec::with_capacity(number_threads);
        ring.resize_with(number_threads, || sync::Mutex::new(vec!()));

        sync::Arc::new(ring)
    };

    let pool = Pool::new(number_threads, |thread_index| {
        let allocation = allocation.clone();
        let shuffle_start = shuffle_start.clone();
        let shuffle_end = shuffle_end.clone();
        let deallocation = deallocation.clone();
        let next = next.clone();
        let ring = ring.clone();

        move || {
            LL_ALLOCATOR.warm_up().expect("Warmed up");

            let custodian = thread_index == 0;

            for iteration in 0..number_iterations {
                //  Prepare batch of victims
                let victims = create_victims(number_victims);

                //  Move victims to the vector of Pointers, which requires allocation.
                {
                    //  Pre-acquire guard to avoid delaying the start for unrelated reasons.
                    let mut sink = ring[thread_index].try_lock().unwrap();

                    allocation.wait_until_all_ready();

                    push_victims(victims, &mut *sink);
                }

                //  Rearm next iteration.
                if custodian {
                    next.reset(number_threads);
                }

                shuffle_start.wait_until_all_ready();

                if custodian {
                    allocation.reset(number_threads);
                }

                //  Shuffle the pointers.
                if custodian {
                    shuffle_ring(&*ring);
                }

                shuffle_end.wait_until_all_ready();

                if custodian {
                    shuffle_start.reset(number_threads);
                }

                //  Deallocate the pointers, recover the victims.
                let victims = {
                    //  Pre-acquire guard to avoid delaying the start for unrelated reasons.
                    let mut stream = ring[thread_index].try_lock().unwrap();
                    let mut victims = Vec::with_capacity(stream.len());

                    deallocation.wait_until_all_ready();

                    pop_victims(&mut *stream, &mut victims);

                    victims
                };

                for (index, victim) in victims.into_iter().enumerate() {
                    assert_eq!(Ok(index), victim.parse(),
                        "thread {}, iteration {}, index {}, victim {:?}", thread_index, iteration, index, victim);
                }

                if custodian {
                    shuffle_end.reset(number_threads);
                }

                next.wait_until_all_ready();

                if custodian {
                    deallocation.reset(number_threads);
                }
            }
        }
    });

    pool.join();
}

//
//  Multi-threaded helpers
//

struct Pool<T>(Vec<thread::JoinHandle<T>>);

impl<T> Pool<T> {
    fn new<F, G>(count: usize, mut factory: F) -> Self
        where
            F: FnMut(usize) -> G,
            G: FnOnce() -> T + Send + 'static,
            T: Send + 'static
    {
        let threads : Vec<_> = (0..count)
            .map(|i| {
                thread::spawn(factory(i))
            })
            .collect();

        Self(threads)
    }

    fn join(mut self) -> Vec<T> {
        let thread_handles = std::mem::replace(&mut self.0, vec!());
        Self::join_handles(thread_handles)
    }

    fn join_handles(thread_handles: Vec<thread::JoinHandle<T>>) -> Vec<T> {
        //  First join _all_ threads.
        let results: Vec<_> = thread_handles.into_iter()
            .map(|handle| handle.join())
            .collect();
        //  Then collect the results.
        results.into_iter()
            .map(|value| value.unwrap())
            .collect()
    }
}

impl<T> Drop for Pool<T> {
    fn drop(&mut self) {
        let thread_handles = std::mem::replace(&mut self.0, vec!());
        Self::join_handles(thread_handles);
    }
}

//  #   Warning
//
//  Rearming a RendezVous is complicated:
//
//  1.  An instance should only be rearmed by 1 thread.
//  2.  An instance cannot be rearmed right before a `wait_until_all_ready` on the same instance.
//  3.  An instance cannot be rearmed right after a `wait_until_all_ready` on the same instance.
//
//  Rearming before is incorrect:
//
//  ```rust
//  if custodian {
//      barrier.reset(x);
//  }
//
//  barrier.wait_until_all_ready();     //  Some threads may already have decremented the counter prior to the reset.
//  ```
//
//  Rearming after is incorrect:
//
//  ```rust
//  barrier.wait_until_all_ready();
//
//  if custodian {
//      barrier.reset(x);               //  Not all threads may have exited the wait, and those who didn't now get stuck.
//  }
//  ```
#[derive(Clone, Debug)]
struct RendezVous(&'static str, sync::Arc<atomic::AtomicUsize>);

impl RendezVous {
    fn new(name: &'static str, count: usize) -> Self {
        Self(name, sync::Arc::new(atomic::AtomicUsize::new(count)))
    }

    fn wait_until_all_ready(&self) {
        self.1.fetch_sub(1, atomic::Ordering::AcqRel);

        while !self.is_ready() {}
    }

    fn is_ready(&self) -> bool { self.1.load(atomic::Ordering::Acquire) == 0 }

    fn reset(&self, count: usize) {
        assert!(self.is_ready(), "{} not ready: {:?}", self.0, self.1);
        self.1.store(count, atomic::Ordering::Release);
    }
}

//
//  Implementation Details
//

fn number_iterations() -> usize { read_number_from_environment("LLMALLOC_MULTI_NUMBER_ITERATIONS", 10) }

fn number_threads() -> usize { read_number_from_environment("LLMALLOC_MULTI_NUMBER_THREADS", 4) }

fn read_number_from_environment(name: &str, default: usize) -> usize {
    for (n, value) in std::env::vars() {
        if n == name {
            if let Ok(result) = value.parse() {
                println!("read_number_from_environment - {}: {}", name, result);
                return result;
            }
        }
    }

    println!("read_number_from_environment - {}: {} (default)", name, default);
    default
}

struct Pointer<T> {
    pointer: *mut T,
}

impl<T> Pointer<T> {
    fn new(value: T) -> Self {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let pointer = LL_ALLOCATOR.allocate(layout(size, align)) as *mut T;

        unsafe { std::ptr::write(pointer, value) }

        Pointer { pointer }
    }

    fn replace_with_default(&mut self) -> T
        where
            T: Default
    {
        std::mem::replace(&mut *self, T::default())
    }
}

impl<T> Default for Pointer<T>
    where
        T: Default
{
    fn default() -> Self { Self::new(T::default()) }
}

impl<T> Drop for Pointer<T> {
    fn drop(&mut self) {
        unsafe {
            std::ptr::drop_in_place(self.pointer);
            LL_ALLOCATOR.deallocate(self.pointer as *mut u8);
        }
    }
}

impl<T> ops::Deref for Pointer<T> {
    type Target = T;

    fn deref(&self) -> &T { unsafe { &*self.pointer } }
}

impl<T> ops::DerefMut for Pointer<T> {
    fn deref_mut(&mut self) -> &mut T { unsafe { &mut *self.pointer} }
}

unsafe impl<T> Send for Pointer<T>
    where
        T: Send
{}

fn layout(size: usize, alignment: usize) -> Layout {
    Layout::from_size_align(size, alignment).expect("Valid Layout")
}

//  FIXME: use sys crates... properly configured for system libraries.
#[link(name = "numa")]
extern "C" {}
