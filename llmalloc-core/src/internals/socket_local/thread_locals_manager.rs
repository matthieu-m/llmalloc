//! Manager of Thread Locals.

use core::{
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ptr::{self, NonNull},
    sync::atomic::{self, Ordering},
};

use crate::{Configuration, PowerOf2};
use crate::{
    internals::{
        atomic_stack::{AtomicStack, AtomicStackElement, AtomicStackLink},
        thread_local::ThreadLocal,
    },
    utils,
};

//  Manager of Thread Locals.
pub(crate) struct ThreadLocalsManager<C> {
    //  Owner.
    owner: *mut (),
    //  Stack of available thread-locals.
    //
    //  AtomicStack is not entirely ABA-proof in case of concurrent pop vs pop+re-push. Here, it should work fine as
    //  the delay between pop and re-push is large, considering that it involves creating _and_ destroying an OS thread.
    //
    //  In a low-latency scenario, with uninterrupted threads, the chances of _that_ happening within the time another
    //  thread takes to just execute pop are exceedingly low, and thus _hopefully_ the merkle-chain of AtomicStack will
    //  be sufficient to guard against those rare cases.
    stack: AtomicStack<MaybeThreadLocal<C>>,
    //  Current watermark for fresh allocations into the buffer area.
    watermark: atomic::AtomicPtr<u8>,
    //  Begin of buffer area.
    begin: NonNull<u8>,
    //  End of buffer area.
    end: NonNull<u8>,
    //  Marker...
    _configuration: PhantomData<*const C>,
}

impl<C> ThreadLocalsManager<C> 
    where
        C: Configuration,
{
    const THREAD_LOCAL_SIZE: usize = mem::size_of::<GuardedThreadLocal<C>>();

    //  Creates an instance which will carve-up the memory of `place` into `ThreadLocals`.
    pub(crate) fn new(owner: *mut (), buffer: &mut [u8]) -> Self {
        let _configuration = PhantomData;

        //  Safety:
        //  -   `buffer.len()` is valid if `buffer` is valid.
        let end = unsafe { NonNull::new_unchecked(buffer.as_mut_ptr().add(buffer.len())) };
        debug_assert!(utils::is_sufficiently_aligned_for(end, PowerOf2::align_of::<GuardedThreadLocal<C>>()));

        let nb_thread_locals = buffer.len() / Self::THREAD_LOCAL_SIZE;
        debug_assert!(nb_thread_locals * Self::THREAD_LOCAL_SIZE <= buffer.len());

        //  Safety:
        //  -   `watermark` still points inside `buffer`, as `x / y * y <= x`.
        let start = unsafe { end.as_ptr().sub(nb_thread_locals * Self::THREAD_LOCAL_SIZE) };
        let watermark = atomic::AtomicPtr::new(start);

        //  Safety:
        //  -   `start` is not null, since `watermark` is not null.
        let begin = unsafe { NonNull::new_unchecked(start) };

        let stack = AtomicStack::default();

        Self { owner, stack, watermark, begin, end, _configuration, }
    }

    //  Acquires a ThreadLocal, if possible.
    pub(crate) fn acquire(&self) -> Option<NonNull<ThreadLocal<C>>> {
        const RELAXED: Ordering = Ordering::Relaxed;

        //  Pick one from stack, if any.
        if let Some(thread_local) = self.pop() {
            return Some(thread_local);
        }

        let mut current = self.watermark.load(RELAXED);
        let end = self.end.as_ptr();

        while current < end {
            //  Safety:
            //  -   `next` still within original `buffer`, as `current < end`.
            let next = unsafe { current.add(Self::THREAD_LOCAL_SIZE) };

            match self.watermark.compare_exchange(current, next, RELAXED, RELAXED) {
                Ok(_) => break,
                Err(previous) => current = previous,
            }
        }

        //  Acquisition failed.
        if current == end {
            return None;
        }

        //  Safety:
        //  -   `current` is not null.
        let current = unsafe { NonNull::new_unchecked(current) };

        debug_assert!(utils::is_sufficiently_aligned_for(current, PowerOf2::align_of::<GuardedThreadLocal<C>>()));

        #[allow(clippy::cast_ptr_alignment)]
        let current = current.as_ptr() as *mut GuardedThreadLocal<C>;

        //  Safety:
        //  -   `current` is valid for writes.
        //  -   `current` is properly aligned.
        unsafe { ptr::write(current, GuardedThreadLocal::new(self.owner)) };

        //  Safety:
        //  -   `current` is non null.
        let guarded = unsafe { &*current };

        Some(NonNull::from(unsafe { &*guarded.maybe_thread_local.thread_local }))
    }

    //  Releases a ThreadLocal, after use.
    //
    //  #   Safety
    //
    //  -   Assumes that `thread_local` points to a valid memory area.
    //  -   Assumes that `thread_local` has exclusive access to this memory area.
    pub(crate) unsafe fn release(&self, thread_local: NonNull<ThreadLocal<C>>) {
        debug_assert!(self.begin <= thread_local.cast());
        debug_assert!(thread_local.cast() < self.end);

        self.push(thread_local);
    }

    //  Internal; Pops a ThreadLocal off the stack, if any.
    fn pop(&self) -> Option<NonNull<ThreadLocal<C>>> {
        self.stack.pop().map(|maybe| {
            //  Safety:
            //  -   Access to `maybe` is exclusive.
            unsafe { MaybeThreadLocal::into_thread_local(maybe, self.owner) }
        })
    }

    //  Internal; Pushes a ThreadLocal onto the stack.
    unsafe fn push(&self, thread_local: NonNull<ThreadLocal<C>>) {
        let maybe = MaybeThreadLocal::from_thread_local(thread_local);

        self.stack.push(&mut *maybe.as_ptr());
    }
}

//
//  Implementation
//

//  A simple padding wrapper to avoid false-sharing between `ThreadLocal`.
#[repr(C)]
struct GuardedThreadLocal<C>{
    _guard: utils::PrefetchGuard,
    maybe_thread_local: MaybeThreadLocal<C>,
}

impl<C> GuardedThreadLocal<C>
    where
        C: Configuration
{
    fn new(owner: *mut ()) -> Self {
        GuardedThreadLocal { _guard: Default::default(), maybe_thread_local: MaybeThreadLocal::new(owner) }
    }
}

impl<C> Default for GuardedThreadLocal<C>
    where
        C: Configuration
{
    fn default() -> Self { Self::new(ptr::null_mut()) }
}

#[repr(align(128))]
union MaybeThreadLocal<C> {
    next: ManuallyDrop<AtomicStackLink<Self>>,
    thread_local: ManuallyDrop<ThreadLocal<C>>,
}

impl<C: Configuration> MaybeThreadLocal<C> {
    fn new(owner: *mut ()) -> Self {
        Self { thread_local: ManuallyDrop::new(ThreadLocal::new(owner)) }
    }

    //  Returns `pointer` memory with freshly initialized `AtomicStackLink` instance inside.
    //
    //  #   Safety:
    //
    //  -   Assumes exclusive access to the memory pointed to by `this`.
    unsafe fn from_thread_local(pointer: NonNull<ThreadLocal<C>>) -> NonNull<Self> {
        let mut this: NonNull<Self> = pointer.cast();

        this.as_mut().next = ManuallyDrop::new(AtomicStackLink::default());

        this
    }

    //  Returns `this` memory with freshly initialized `ThreadLocal` instance inside.
    //
    //  #   Safety:
    //
    //  -   Assumes exclusive access to the memory pointed to by `this`.
    unsafe fn into_thread_local(mut this: NonNull<Self>, owner: *mut ()) -> NonNull<ThreadLocal<C>> {
        let this = this.as_mut();

        this.thread_local = ManuallyDrop::new(ThreadLocal::new(owner));

        NonNull::from(&mut *this.thread_local)
    }
}

impl<C> AtomicStackElement for MaybeThreadLocal<C> {
    fn next(&self) -> &AtomicStackLink<Self> { unsafe { &self.next } }
}

#[cfg(test)]
mod tests {

use core::slice;

use super::*;
use super::super::test::TestConfiguration;

type TestThreadLocalsManager = ThreadLocalsManager<TestConfiguration>;

#[repr(align(128))]
struct ThreadLocalsStore([u8; 8192]);

impl ThreadLocalsStore {
    const THREAD_LOCAL_SIZE: usize = TestThreadLocalsManager::THREAD_LOCAL_SIZE;

    //  Creates a ThreadLocalsManager.
    //
    //  #   Safety
    //
    //  -   Takes over the memory.
    unsafe fn create(&self) -> TestThreadLocalsManager {
        let buffer = slice::from_raw_parts_mut(self.0.as_ptr() as *mut _, self.0.len());

        ThreadLocalsManager::new(ptr::null_mut(), buffer)
    }
}

impl Default for ThreadLocalsStore {
    fn default() -> Self { unsafe { mem::zeroed() } }
}

#[test]
fn thread_locals_manager_new() {
    let store = ThreadLocalsStore::default();
    let manager = unsafe { store.create() };

    let watermark = manager.watermark.load(Ordering::Relaxed);

    assert_eq!(None, manager.stack.pop());
    assert_ne!(ptr::null_mut(), watermark);

    let bytes = manager.end.as_ptr() as usize - watermark as usize;

    assert_eq!(7680, bytes);
    assert_eq!(0, bytes % ThreadLocalsStore::THREAD_LOCAL_SIZE);
    assert_eq!(10, bytes / ThreadLocalsStore::THREAD_LOCAL_SIZE);
}

#[test]
fn thread_locals_acquire_release() {
    let store = ThreadLocalsStore::default();
    let manager = unsafe { store.create() };

    //  Acquire fresh pointers, by bumping the watermark.
    let mut thread_locals = [ptr::null_mut(); 10];

    for ptr in &mut thread_locals {
        *ptr = manager.acquire().unwrap().as_ptr();
    }

    //  Watermark bumped all the way through.
    let watermark = manager.watermark.load(Ordering::Relaxed);
    assert_eq!(manager.end.as_ptr(), watermark);

    //  No more!
    assert_eq!(None, manager.acquire());

    //  Release thread-locals.
    for ptr in &thread_locals {
        unsafe { manager.release(NonNull::new(*ptr).unwrap()) };
    }

    //  Acquire them again, in reverse order.
    for ptr in thread_locals.iter().rev() {
        assert_eq!(*ptr, manager.acquire().unwrap().as_ptr());
    }
}

} // mod tests
