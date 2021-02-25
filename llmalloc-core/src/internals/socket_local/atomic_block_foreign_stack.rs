//! A thread-safe stack of Foreign Blocks of memory.
//!
//! Writing a wait-free, or even lock-free, stack is a tad complicated. Luckily for us, we only need it to manage the
//! List of thread-local data-structures, so we're not even going to try: creating or destroying an OS thread are such
//! heavy-weight synchronized operations that there's no point in even attempting it.

use core::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::internals::blocks::BlockForeign;

#[derive(Default)]
pub(crate) struct AtomicBlockForeignStack(Mutex<Option<NonNull<BlockForeign>>>);

impl AtomicBlockForeignStack {
    /// Pops the top of the stack, if any.
    pub(crate) fn pop(&self) -> Option<NonNull<BlockForeign>> {
        let mut guard = self.0.lock();

        if let Some(head_ptr) = guard.take() {
            //  Safety:
            //  -   `head_ptr` points to a valid `BlockForeign`.
            //  -   `head_ptr` has exclusive access to this `BlockForeign`. 
            let head = unsafe { head_ptr.as_ref() };

            *guard = head.next.replace_with_null();

            Some(head_ptr)
        } else {
            None
        }
    }

    /// Pushes onto the stack.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `block` points to a valid `BlockForeign` to which it has exclusive access for as long as it
    ///     remains in the stack.
    pub(crate) unsafe fn push(&self, block: NonNull<BlockForeign>) {
        let mut guard = self.0.lock();

        {
            //  Safety:
            //  -   `block` points to a valid `BlockForeign`.
            //  -   `block` has exclusive access to this `BlockForeign`. 
            let block = block.as_ref();

            block.next.set(guard.take());
        }

        *guard = Some(block);
    }

    //  Test-only
    #[cfg(test)]
    fn len(&self) -> usize {
        let guard = self.0.lock();

        let mut len = 0;

        let mut head = guard.clone();
        while let Some(inner) = head {
            len += 1;
            //  Safety:
            //  -   `block` points to a valid `BlockForeign`.
            //  -   `block` has exclusive access to this `BlockForeign`. 
            head = unsafe { inner.as_ref() }.next.get();
        }

        len
    }
}

unsafe impl Send for AtomicBlockForeignStack {}

//
//  Implementation Details
//

const UNLOCKED: usize = 0;
const LOCKED: usize = !0usize;

#[derive(Default)]
struct Mutex<T> {
    mutex: AtomicUsize,
    element: UnsafeCell<T>,
}

struct MutexGuard<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<T> Mutex<T> {
    fn lock<'a>(&'a self) -> MutexGuard<'a, T> {
        while let Err(_) = self.mutex.compare_exchange(UNLOCKED, LOCKED, Ordering::Acquire, Ordering::Relaxed) {}

        MutexGuard { mutex: self, }
    }

    fn unlock(&self) {
        debug_assert!(self.mutex.load(Ordering::Relaxed) == LOCKED);
        self.mutex.store(UNLOCKED, Ordering::Release);
    }
}

impl<'a, T> Deref for MutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T { unsafe { &*self.mutex.element.get() } }
}

impl<'a, T> DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T { unsafe { &mut *self.mutex.element.get() } }
}

impl<'a, T> Drop for MutexGuard<'a, T> {
    fn drop(&mut self) {
        self.mutex.unlock();
    }
}

unsafe impl<T: Send> Send for Mutex<T> {}
unsafe impl<T> Sync for Mutex<T> {}

#[cfg(test)]
mod tests {

use llmalloc_test::BurstyBuilder;

use super::*;

type Stack = AtomicBlockForeignStack;

#[test]
fn atomic_block_foreign_stack_sync() {
    fn ensure_send<T: Send>() {}
    fn ensure_sync<T: Sync>() {}

    ensure_send::<Stack>();
    ensure_sync::<Stack>();
}

#[test]
fn atomic_block_foreign_stack_pop_push() {
    let array = [BlockForeign::default(), BlockForeign::default()];
    let (a, b) = (NonNull::from(&array[0]), NonNull::from(&array[1]));

    let stack = Stack::default();

    assert_eq!(None, stack.pop());
    assert_eq!(0, stack.len());

    unsafe {
        stack.push(a);
        stack.push(b);
    }

    assert_eq!(2, stack.len());

    assert_eq!(Some(b), stack.pop());
    assert_eq!(Some(a), stack.pop());
    assert_eq!(None, stack.pop());
    assert_eq!(0, stack.len());
}

#[test]
fn atomic_block_foreign_stack_concurrent_push_concurrent_pop_fuzzing() {
    //  The test aims at validating that:
    //  -   Multiple threads can push concurrently.
    //  -   Multiple threads can pop concurrently.
    //
    //  To do so:
    //  -   Each thread is given one element.
    //  -   Each thread will repeatedly push this element on the stack, then pop a random element.
    //  -   Each thread will assert that they did manage to pop an element.
    //  -   At the end of the test, the stack should be empty.
    #[derive(Default)]
    struct Local(BlockForeign);

    //  Safety:
    //  -   Guaranteed to have exclusive access to its `element`.
    unsafe impl Send for Local {}

    let elements = vec!(Local::default(), Local::default(), Local::default(), Local::default());

    let mut builder = BurstyBuilder::new(Stack::default(), elements);

    //  Step 1: Push.
    builder.add_simple_step(|| |stack: &Stack, element: &mut Local| {
        unsafe { stack.push(NonNull::from(&element.0)) };
    });

    //  Step 2: Pop one of the pushed elements.
    builder.add_simple_step(|| |stack: &Stack, _: &mut Local| {
        let element = stack.pop();
        assert_ne!(None, element);
    });

    //  Step 3: There should be nothing to pop.
    builder.add_simple_step(|| |stack: &Stack, _: &mut Local| {
        let element = stack.pop();
        assert_eq!(None, element);
    });

    builder.launch(100);
}

#[test]
fn atomic_block_foreign_stack_concurrent_push_pop_fuzzing() {
    //  The test aims at validating that multiple threads can push _and_ pop concurrently.
    //
    //  To do so:
    //  -   Each thread is given an element.
    //  -   In the first step, even threads attempt to push, whilst odd threads attempt to pop.
    //  -   In the second step, the roles are reversed.
    //  -   Each thread assert that they did manage to pop an element.
    //  -   The stack must therefore be primed with the elements from the odd threads, as they pop first.
    #[derive(Default)]
    struct Local {
        index: usize,
        element: Option<NonNull<BlockForeign>>,
    }

    impl Local {
        fn is_pop_then_push(&self) -> bool { self.index % 2 != 0 }
    }

    //  Safety:
    //  -   Guaranteed to have exclusive access to its `element`.
    unsafe impl Send for Local {}

    let store = vec!(BlockForeign::default(), BlockForeign::default(), BlockForeign::default(), BlockForeign::default());

    let elements: Vec<_> = store.iter()
        .enumerate()
        .map(|(index, element)| {
            Local { index, element: Some(NonNull::from(element)), }
        })
        .collect();

    let number_threads = elements.len();
    let half_threads = number_threads / 2;
    assert_eq!(number_threads, half_threads * 2);

    let mut builder = BurstyBuilder::new(Stack::default(), elements);

    //  Step 1: Push elements from half of the threads, ready to pop.
    builder.add_simple_step(|| |stack: &Stack, local: &mut Local| {
        if local.is_pop_then_push() {
            unsafe { stack.push(local.element.unwrap()) };
        }
    });

    //  Step 1.5: Checking stack state.
    builder.add_simple_step(|| move |stack: &Stack, _: &mut Local| {
        assert_eq!(half_threads, stack.len());
    });

    //  Step 2: Pop/Push from half of the threads.
    builder.add_simple_step(|| |stack: &Stack, local: &mut Local| {
        if local.is_pop_then_push() {
            local.element = stack.pop();
            assert_ne!(None, local.element);
        } else {
            unsafe { stack.push(local.element.unwrap()) };
        }
    });

    //  Step 2.5: Checking stack state.
    builder.add_simple_step(|| move |stack: &Stack, _: &mut Local| {
        assert_eq!(half_threads, stack.len());
    });

    //  Step 3: Pop/Push from the other half of the threads.
    builder.add_simple_step(|| |stack: &Stack, local: &mut Local| {
        if local.is_pop_then_push() {
            unsafe { stack.push(local.element.unwrap()) };
        } else {
            local.element = stack.pop();
            assert_ne!(None, local.element);
        }
    });

    //  Step 3.5: Checking stack state.
    builder.add_simple_step(|| move |stack: &Stack, _: &mut Local| {
        assert_eq!(half_threads, stack.len());
    });

    //  Step 4: Drain from the early pushers.
    builder.add_simple_step(|| |stack: &Stack, local: &mut Local| {
        if local.is_pop_then_push() {
            local.element = stack.pop();
            assert_ne!(None, local.element);
        }
    });

    //  Step 5: Ensure the stack is drained.
    builder.add_simple_step(|| |stack: &Stack, _: &mut Local| {
        let element = stack.pop();
        assert_eq!(None, element);
        
        assert_eq!(0, stack.len());
    });

    builder.launch(100);
}

} // mod tests
