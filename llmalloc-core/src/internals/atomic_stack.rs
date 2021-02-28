//! A wait-free intrusively-linked stack.
//!
//! #   Warning
//!
//! Writing a proper wait-free stack is actually far more difficult than one would expect in the absence of
//! [Load-Link/Store-Conditional](https://en.wikipedia.org/wiki/Load-link/store-conditional) or Double-CAS due to
//! ABA issues around manipulation of the top pointer.
//!
//! This stack attempts to circumvent the issue using a regular CAS by using a Merkle-Chain, however this is definitely
//! not foolproof, and the quality of it depends on the platform and the alignment of the linked elements.
//!
//! For best protection:
//!
//! -   Use on platforms with as many "free" top-bits as possible; for example x64 has 17 free top bits in user-space.
//! -   Use the highest alignment possible, to gain even more bits.
//! -   Use only in situations when popping-pushing the same element in quick sequence is unlikely.
//!
//! #   Safety
//!
//! The stack assumes that the elements stashed within are:
//!
//! -   Exclusively accessible through the stack; ie, the stack _owns_ them.
//! -   Will outlive the stack.
//!
//! See the signature of `push`.

use core::{
    marker::PhantomData,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::PowerOf2;

/// AtomicStackElement
pub(crate) trait AtomicStackElement : Sized {
    /// Alignment of the element.
    const ALIGNMENT: PowerOf2 = PowerOf2::align_of::<Self>();

    /// Next link.
    fn next(&self) -> &AtomicStackLink<Self>;
}

/// AtomicStack
pub(crate) struct AtomicStack<T>(AtomicStackLink<T>);

impl<T> AtomicStack<T> {
    /// Creates an empty instance of the AtomicStack.
    pub(crate) fn new() -> Self { Self(AtomicStackLink::default()) }

    /// Checks whether the stack is empty, or not.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool { self.0.is_null() }
}

impl<T: AtomicStackElement> AtomicStack<T> {
    /// Pops the top element, if any.
    ///
    /// The returned `NonNull`, if any, is guaranteed to have exclusive access to its pointee.
    pub(crate) fn pop(&self) -> Option<NonNull<T>> {
        //  WARNING:
        //
        //  Due to concurrency, another thread may start using the data pointed to be `head` prior to this call
        //  terminating.
        //
        //  DO NOT WRITE to `head`/`current` before having exclusive ownership of it.
        let mut current = self.0.load();

        loop {
            //  Safety:
            //  -   `current` was obtained from `Self::pack` or is 0.
            let head = unsafe { Self::unpack(current) }?;

            //  Safety:
            //  -   `head` is a valid pointer, to a valid `T`.
            //
            //  There is a degree of uncertainy as to whether `head.as_ref()` is fine, in isolation. There is no
            //  guarantee that a mutable reference to `*head` doesn't exist at this point, so technically it may
            //  be invalid to form a `&T`, even though in practice we only access an atomic field.
            let next = unsafe { head.as_ref().next().load() };

            if let Err(new_current) = self.0.compare_exchange_weak(current, next) {
                current = new_current;
                continue;
            }

            //  Safety:
            //  -   `head` is a valid pointer, to a valid `T`.
            unsafe { head.as_ref().next().store(0) };

            return Some(head);
        }
    }

    /// Pushes an element at the top.
    pub(crate) fn push<'a, 'b>(&'a self, element: &'b mut T)
        where
            'b: 'a,
    {
        //  WARNING:
        //
        //  Due to concurrency, another thread may start using the data pointed to be `head` prior to this call
        //  terminating.
        //
        //  DO NOT WRITE to `head`/`current`.
        let element = NonNull::from(element);

        let mut current = self.0.load();

        loop {
            //  Safety:
            //  -   `element` is valid.
            //  -   Lifetime is bound.
            unsafe { element.as_ref() }.next().store(current);

            let hash = fnv(current);
            let packed = Self::pack(element, hash);

            if let Err(new_current) = self.0.compare_exchange_weak(current, packed) {
                current = new_current;
                continue;
            }

            break;
        }
    }

    //  Internal.
    //  FIXME: `const` once rustc grows up.
    fn merkle_bits() -> usize {
        T::ALIGNMENT.value().trailing_zeros() as usize + TOP_POINTER_BITS
    }

    //  Internal.
    fn pack(pointer: NonNull<T>, next: usize) -> usize {
        //  FIXME: `const` once rustc grows up.
        let merkle_mask: usize = (1usize << Self::merkle_bits()) - 1;

        let pointer = (pointer.as_ptr() as usize) << TOP_POINTER_BITS;

        pointer | (next & merkle_mask)
    }

    //  Internal.
    //
    //  #   Safety:
    //
    //  -   Assumes that `element` was obtained by `pack`, or is 0.
    unsafe fn unpack(element: usize) -> Option<NonNull<T>> {
        //  FIXME: `const` once rustc grows up.
        let merkle_mask: usize = (1usize << Self::merkle_bits()) - 1;

        if element == 0 {
            return None;
        }

        let pointer = (element & !merkle_mask) >> TOP_POINTER_BITS;
        debug_assert_ne!(0, pointer, "element: {}, merkle: {}, top: {}", element, Self::merkle_bits(), TOP_POINTER_BITS);

        //  Safety:
        //  -   `element` was obtained by packing a pointer.
        NonNull::new(pointer as *mut T)
    }
}

impl<T> Default for AtomicStack<T> {
    fn default() -> Self { Self::new() }
}

/// AtomicStackLink
pub(crate) struct AtomicStackLink<T>(AtomicUsize, PhantomData<T>);

impl<T> AtomicStackLink<T> {
    fn new() -> Self { Self(AtomicUsize::new(0), PhantomData) }

    #[cfg(test)]
    fn is_null(&self) -> bool { self.0.load(Ordering::Relaxed) == 0 }

    fn load(&self) -> usize { self.0.load(Ordering::Acquire) }

    fn store(&self, payload: usize) { self.0.store(payload, Ordering::Release) }

    fn compare_exchange_weak(&self, current: usize, value: usize) -> Result<usize, usize> {
        self.0.compare_exchange_weak(current, value, Ordering::AcqRel, Ordering::Acquire)
    }
}

impl<T> Default for AtomicStackLink<T> {
    fn default() -> Self { Self::new() }
}

//
//  Implementation
//

#[cfg(target_arch = "x86_64")]
const TOP_POINTER_BITS: usize = 17;

#[cfg(not(target_arch = "x86_64"))]
const TOP_POINTER_BITS: usize = 0;

fn fnv(data: usize) -> usize {
    let mut hash: usize = 0xcbf29ce484222325;

    for i in 0..7 {
        hash = hash.wrapping_mul(0x100000001b3);
        hash ^= (data >> (8 * i)) & 0xFF;
    }

    hash
}

#[cfg(test)]
mod tests {

use std::mem;

use llmalloc_test::BurstyBuilder;

use super::*;

type Stack = AtomicStack<Element>;

#[repr(align(128))]
#[derive(Default)]
struct Element {
    next: AtomicStackLink<Element>,
}

impl AtomicStackElement for Element {
    fn next(&self) -> &AtomicStackLink<Self> { &self.next }
}

#[test]
fn atomic_stack_send_sync() {
    fn ensure_send<T: Send>() {}
    fn ensure_sync<T: Sync>() {}

    ensure_send::<Stack>();
    ensure_sync::<Stack>();
}

#[test]
fn atomic_stack_pop_push() {
    let array = [Element::default(), Element::default()];
    let (a, b) = (NonNull::from(&array[0]), NonNull::from(&array[1]));

    let stack = Stack::default();

    assert_eq!(None, stack.pop());
    assert!(stack.is_empty());

    unsafe {
        stack.push(&mut *a.as_ptr());
        stack.push(&mut *b.as_ptr());
    }

    assert!(!stack.is_empty());

    assert_eq!(Some(b), stack.pop());
    assert_eq!(Some(a), stack.pop());
    assert_eq!(None, stack.pop());
    assert!(stack.is_empty());
}

#[test]
fn atomic_stack_concurrent_push_concurrent_pop_fuzzing() {
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
    struct Local(Element);

    //  Safety:
    //  -   Guaranteed to have exclusive access to its `element`.
    unsafe impl Send for Local {}

    let elements = vec!(Local::default(), Local::default(), Local::default(), Local::default());

    let mut builder = BurstyBuilder::new(Stack::default(), elements);

    //  Step 1: Push.
    builder.add_simple_step(|| |stack: &Stack, local: &mut Local| {
        //  Safety:
        //  -   Let's not access stack elements after Local dies, eh?
        stack.push(unsafe { mem::transmute(&mut local.0) });
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
fn atomic_stack_concurrent_push_pop_fuzzing() {
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
        element: Option<NonNull<Element>>,
    }

    impl Local {
        fn is_pop_then_push(&self) -> bool { self.index % 2 != 0 }
    }

    //  Safety:
    //  -   Guaranteed to have exclusive access to its `element`.
    unsafe impl Send for Local {}

    let store = vec!(Element::default(), Element::default(), Element::default(), Element::default());

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
            //  Safety:
            //  -   I promise to try not to access stack elements after `local` has died...
            stack.push(unsafe { &mut *local.element.unwrap().as_ptr() });
        }
    });

    //  Step 2: Pop/Push from half of the threads.
    builder.add_simple_step(|| |stack: &Stack, local: &mut Local| {
        if local.is_pop_then_push() {
            local.element = stack.pop();
            assert_ne!(None, local.element);
        } else {
            //  Safety:
            //  -   I promise to try not to access stack elements after `local` has died...
            stack.push(unsafe { &mut *local.element.unwrap().as_ptr() });
        }
    });

    //  Step 3: Pop/Push from the other half of the threads.
    builder.add_simple_step(|| |stack: &Stack, local: &mut Local| {
        if local.is_pop_then_push() {
            //  Safety:
            //  -   I promise to try not to access stack elements after `local` has died...
            stack.push(unsafe { &mut *local.element.unwrap().as_ptr() });
        } else {
            local.element = stack.pop();
            assert_ne!(None, local.element);
        }
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
        
        assert!(stack.is_empty());
    });

    builder.launch(100);
}    

} // mod tests
