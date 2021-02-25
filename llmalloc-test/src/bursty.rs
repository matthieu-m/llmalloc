//! A test-runner for detecting data-races and race-conditions.

use std::{
    cell::RefCell,
    sync::{Arc, atomic::{AtomicIsize, Ordering}},
    thread::{self, JoinHandle},
};

/// Bursty is a test-runner for writing tests specializing in flushing out data-races and race-conditions.
///
/// Bursty is a multi-thread coordinator to run user-specified steps _in lockstep_ across multiple threads of execution.
///
/// Bursty allows the user to:
///
/// -   Register a Global state, shared across all threads.
/// -   Register N instances of a Local state, each dedicated to a single thread.
/// -   Register S steps, which will run on each thread, in lock-step with other threads.
///
/// In particular, the specialty of Bursty is to ensure that the step Si starts as simultaneously as possible on each
/// thread.
///
/// Constructing a `Bursty` is done through a `BurstyBuilder`.
pub struct Bursty<Global, Local> {
    global: Arc<Global>,
    threads: RefCell<Vec<JoinHandle<Local>>>,
    results: RefCell<Vec<Local>>,
}

impl<Global, Local> Bursty<Global, Local> {
    /// Creates an instance of Bursty.
    pub fn new(global: Arc<Global>, threads: Vec<JoinHandle<Local>>) -> Self {
        assert!(!threads.is_empty());

        let threads = RefCell::new(threads);
        let results = RefCell::new(vec!());

        Self { global, threads, results, }
    }

    /// Join the threads, and collects their results.
    ///
    /// #   Panics
    ///
    /// -   If any of the threads being joined panicked.
    pub fn join(&self) {
        if !self.results.borrow().is_empty() {
            return;
        }

        let results = self.threads.replace(vec!())
            .drain(..)
            .map(|handle| handle.join().unwrap())
            .collect();

        self.results.replace(results);
    }

    /// Returns a reference to the Global state.
    ///
    /// #   Warning
    ///
    /// Access is provided _without_ joining the threads first.
    pub fn global(&self) -> &Global { &*self.global }

    /// Returns a clone of the Local state.
    ///
    /// Calls `self.join()` to collect them first, if not already done.
    ///
    /// #   Panic
    ///
    /// -   If `self.join()` panics.
    /// -   If cloning `Vec<Local>` panics.
    pub fn locals(&self) -> Vec<Local>
        where
            Local: Clone,
    {
        self.join();

        self.results.borrow().clone()
    }
}

impl<Global, Local> Drop for Bursty<Global, Local> {
    fn drop(&mut self) {
        self.join();
    }
}

/// BurstyBuilder, a builder for a `Bursty` instance.
///
/// #   Example
///
/// A simple demonstration of constructing an instance `Bursty`.
///
/// ```
/// use std::sync::atomic::{AtomicI32, Ordering};
/// use llmalloc_test::BurstyBuilder;
///
/// let mut builder = BurstyBuilder::new(AtomicI32::new(0), vec!(1, 10));
///
/// builder.add_simple_step(|| |global: &AtomicI32, local: &mut i32| { global.fetch_add(*local, Ordering::Relaxed); });
///
/// let bursty = builder.launch(4);
///
/// assert!(44 >= bursty.global().load(Ordering::Relaxed));
///
/// bursty.join();
///
/// assert_eq!(44, bursty.global().load(Ordering::Relaxed));
/// assert_eq!(vec!(1, 10), bursty.locals());
/// ```
pub struct BurstyBuilder<Global, Local> {
    global: Arc<Global>,
    locals: Vec<Local>,
    steps: Vec<Vec<Box<dyn FnMut(&Global, &mut Local) + Send + 'static>>>,
    rendez_vous: Vec<RendezVous>,
}

impl<Global, Local> BurstyBuilder<Global, Local>
    where
        Global: Send + Sync + 'static,
        Local: Send + 'static,
{
    /// Creates a new instance of BurstyBuilder.
    pub fn new(global: Global, locals: Vec<Local>) -> Self {
        let global = Arc::new(global);
        let steps = {
            let mut steps = vec!();
            steps.resize_with(locals.len(), || vec!());
            steps
        };
        let rendez_vous = vec!(RendezVous::new(locals.len()));

        Self { global, locals, steps, rendez_vous, }
    }

    /// Adds a minimal step on each thread. It accesses no state, neither global nor local.
    ///
    /// The step is created by invoking `factory` for each thread.
    pub fn add_minimal_step<Factory, Step>(&mut self, mut factory: Factory)
        where
            Factory: FnMut() -> Step,
            Step: FnMut() + Send + 'static,
    {
        self.add_simple_step(move || {
            let mut step = factory();
            move |_: &Global, _: &mut Local| step()
        })
    }

    /// Adds a simple step on each thread.
    ///
    /// The step is created by invoking `factory` for each thread.
    pub fn add_simple_step<Factory, Step>(&mut self, mut factory: Factory)
        where
            Factory: FnMut() -> Step,
            Step: FnMut(&Global, &mut Local) + Send + 'static,
    {
        self.add_complex_step(move || {
            let mut step = factory();
            let prep = |_: &Global, _: &mut Local| ();
            let step = move |global: &Global, local: &mut Local, _: ()| step(global, local);
            (prep, step)
        });
    }

    /// Adds a step to each thread, as determined by the number of instances of `Local`.
    ///
    /// The step is split in two, and both are created by invoking `factory` for each thread:
    ///
    /// -   A preparatory step Prep, returning R.
    /// -   The actual step Step.
    ///
    /// The preparatory step Prep is run before waiting for the other threads, and is therefore ideal to run expensive
    /// preparatory work.
    pub fn add_complex_step<Factory, Prep, R, Step>(&mut self, mut factory: Factory)
        where
            Factory: FnMut() -> (Prep, Step),
            Prep: FnMut(&Global, &mut Local) -> R + Send + 'static,
            Step: FnMut(&Global, &mut Local, R) + Send + 'static,
    {
        let rendez_vous = RendezVous::new(self.locals.len());

        for serie in &mut self.steps {
            let rendez_vous = rendez_vous.clone();
            let (mut prep, mut step) = factory();
            let prev = self.rendez_vous.last().unwrap().clone();

            serie.push(Box::new(move |global: &Global, local: &mut Local| {
                let prepared = prep(global, local);

                rendez_vous.wait_until_all_ready();

                step(global, local, prepared);

                prev.reset();
            }));
        }

        self.rendez_vous.push(rendez_vous);
    }

    /// Creates the Bursty instance, which will run each serie of steps `iterations` times.
    ///
    /// The threads start immediately.
    pub fn launch(mut self, iterations: usize) -> Bursty<Global, Local> {
        assert!(self.steps.len() > 0,
            "Cannot launch a burst test without a single thread");
        assert!(self.steps[0].len() > 0,
            "Cannot launch a burst test without a single step");

        //  The algorithm used for lock-step only works with a minimum of 3 steps, including the last step added below.
        //
        //  With only 2 steps, the "previous" RendezVous is also the "next" RendezVous, which will cause some threads
        //  to reset it whilst others are already waiting on it.
        //
        //  If there is a single step, then adding the finish step won't meet the requirements, hence we add a dummy
        //  step here. It does nothing but coordinating the synchronization of steps.
        if self.steps[0].len() < 2 {
            self.add_minimal_step(|| || ());
        }

        for serie in &mut self.steps {
            let last = self.rendez_vous.first().unwrap().clone();
            let prev = self.rendez_vous.last().unwrap().clone();

            serie.push(Box::new(move |_: &Global, _: &mut Local| {
                last.wait_until_all_ready();

                prev.reset();
            }));
        }

        assert!(self.steps[0].len() >= 3);

        let mut threads = vec!();
        let rendez_vous = Arc::new(self.rendez_vous);

        for (mut local, mut serie) in self.locals.into_iter().zip(self.steps.into_iter()) {
            let global = self.global.clone();
            let rendez_vous = rendez_vous.clone();

            threads.push(thread::spawn(move || {
                let mut guard = PoisonGuard(rendez_vous);

                let global = &*global;

                for _ in 0..iterations {
                    for step in &mut serie {
                        step(global, &mut local);
                    }
                }

                guard.dismiss();

                local
            }));
        }

        let global = self.global;
        Bursty::new(global, threads)
    }
}

//
//  Implementation details
//

//  If a single thread panics, then we need to abort the execution of all threads.
struct PoisonGuard(Arc<Vec<RendezVous>>);

impl PoisonGuard {
    fn dismiss(&mut self) { self.0 = Arc::default() }
}

impl Drop for PoisonGuard {
    fn drop(&mut self) {
        for rendez_vous in &*self.0 {
            rendez_vous.poison();
        }
    }
}

#[derive(Clone, Debug)]
struct RendezVous(Arc<(AtomicIsize, isize)>);

impl RendezVous {
    fn new(count: usize) -> Self {
        assert!(count <= (isize::MAX as usize));
        Self(Arc::new((AtomicIsize::new(count as isize), count as isize)))
    }

    fn poison(&self) {
        self.0.0.store(-1, Ordering::Relaxed);
    }

    fn wait_until_all_ready(&self) {
        self.0.0.fetch_sub(1, Ordering::Relaxed);

        while !self.is_ready() {}
    }

    fn reset(&self) {
        let mut count = self.load();

        while let Err(_) = self.0.0.compare_exchange(count as isize, self.0.1, Ordering::Relaxed, Ordering::Relaxed) {
            count = self.load();
        }
    }

    //  Internal.
    fn is_ready(&self) -> bool { self.load() == 0 }

    //  Internal.
    fn load(&self) -> usize {
        let count = self.0.0.load(Ordering::Relaxed);

        if count < 0 { self.abandon_ship() }

        count as usize
    }

    //  Internal.
    #[cold]
    #[inline(never)]
    fn abandon_ship(&self) {
        panic!("Someone poisoned the well!");
    }
}

#[cfg(test)]
mod tests {

use std::sync::Mutex;

use super::*;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct LocalEvent {
    iteration: usize,
    step: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
struct GlobalEvent {
    iteration: usize,
    step: usize,
    thread: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct LocalTrace {
    thread: usize,
    events: Vec<LocalEvent>,
}

impl LocalTrace {
    //  Creates the sequence of LocalTrace expected by the constructor of `BurstyBuilder` for the given number of
    //  `threads`.
    fn create(threads: usize) -> Vec<LocalTrace> {
        assert!(threads > 0);

        let mut result = vec!();
        for t in 0..threads {
            result.push(LocalTrace { thread: t, events: vec!(), });
        }
        result
    }

    //  Returns the expected serie of LocalTraces based on the number of threads, iterations, and steps.
    fn expected(threads: usize, iterations: usize, steps: usize) -> Vec<LocalTrace> {
        assert!(iterations > 0);
        assert!(steps > 0);

        let mut result = Self::create(threads);
        for local in &mut result {
            for i in 0..iterations {
                for s in 0..steps {
                    local.add(i, s);
                }
            }
        }
        result
    }

    //  Internal: add a LocalEvent to the recorded trace.
    fn add(&mut self, iteration: usize, step: usize) {
        self.events.push(LocalEvent { step, iteration, });
    }
}

#[derive(Debug, Default)]
struct GlobalTrace {
    events: Mutex<Vec<GlobalEvent>>,
}

impl GlobalTrace {
    //  Creates a `BurstyBuilder` for the given number of threads. 
    fn create_builder(threads: usize) -> BurstyBuilder<GlobalTrace, LocalTrace> {
        assert!(threads > 0);

        BurstyBuilder::new(GlobalTrace::default(), LocalTrace::create(threads))
    }

    //  Returns the expected serie of GlobalEvent based on the number of threads, iterations, and steps.
    fn expected(threads: usize, iterations: usize, steps: usize) -> Vec<GlobalEvent> {
        assert!(threads > 0);
        assert!(iterations > 0);
        assert!(steps > 0);

        let mut result = vec!();

        for i in 0..iterations {
            for s in 0..steps {
                for t in 0..threads {
                    result.push(GlobalEvent{ iteration: i, step: s, thread: t, });
                }
            }
        }

        result
    }

    //  Create a step of index `step`.
    //
    //  This step will record each invocation in the GlobalTrace and LocalTrace, keeping track of the number of
    //  iterations.
    fn create_step(step: usize) -> (impl FnMut(&GlobalTrace, &mut LocalTrace) -> usize, impl FnMut(&GlobalTrace, &mut LocalTrace, usize)) {
        let mut iteration = 0;

        let prep = move |_: &GlobalTrace, _: &mut LocalTrace| {
            let tmp = iteration;
            iteration += 1;
            tmp
        };

        let step = move |global: &GlobalTrace, local: &mut LocalTrace, iteration: usize| {
            global.add(local.thread, iteration, step);
            local.add(iteration, step);
        };

        (prep, step)
    }

    //  Returns the events logged.
    //
    //  To make comparison with expected events easier, each sub-sequence of equal iteration and step is sorted by
    //  thread.
    fn events(&self) -> Vec<GlobalEvent> {
        fn split(slice: &mut [GlobalEvent]) -> (&mut [GlobalEvent], &mut [GlobalEvent]) {
            let front = slice.first().expect("Not empty").clone();

            for (i, e) in slice.iter().enumerate() {
                if e.iteration != front.iteration || e.step != front.step {
                    return slice.split_at_mut(i);
                }
            }

            (slice, &mut [])
        }

        let mut events = self.events.lock()
            .unwrap()
            .clone();

        //  The order in which threads have enqueued events is unpredictable. In order to compare with the `expected`
        //  events, we must re-order every sub-serie by "thread".
        //
        //  At the same time, though, it is critically important NOT to reorder by iteration or step. The goal of
        //  GlobalTrace is to verify that lock-step was correctly handled, after all.
        let mut slice = &mut events[..];

        while !slice.is_empty() {
            let (head, tail) = split(slice);
            slice = tail;

            head.sort()
        }

        events
    }

    //  Internal: Appends a GlobalEvent to the recorded trace.
    fn add(&self, thread: usize, iteration: usize, step: usize) {
        let mut events = self.events.lock().unwrap();
        events.push(GlobalEvent{ iteration, step, thread, });
    }
}

#[test]
fn single_thread_single_step_single_iteration() {
    let mut builder = GlobalTrace::create_builder(1);

    builder.add_complex_step(|| GlobalTrace::create_step(0));

    let bursty = builder.launch(1);

    assert_eq!(LocalTrace::expected(1, 1, 1), bursty.locals());
    assert_eq!(GlobalTrace::expected(1, 1, 1), bursty.global().events());
}

#[test]
fn single_thread_single_step_n_iterations() {
    let mut builder = GlobalTrace::create_builder(1);

    builder.add_complex_step(|| GlobalTrace::create_step(0));

    let bursty = builder.launch(3);

    assert_eq!(LocalTrace::expected(1, 3, 1), bursty.locals());
    assert_eq!(GlobalTrace::expected(1, 3, 1), bursty.global().events());
}

#[test]
fn single_thread_n_steps_single_iteration() {
    let mut builder = GlobalTrace::create_builder(1);

    builder.add_complex_step(|| GlobalTrace::create_step(0));
    builder.add_complex_step(|| GlobalTrace::create_step(1));
    builder.add_complex_step(|| GlobalTrace::create_step(2));
    builder.add_complex_step(|| GlobalTrace::create_step(3));
    builder.add_complex_step(|| GlobalTrace::create_step(4));

    let bursty = builder.launch(1);

    assert_eq!(LocalTrace::expected(1, 1, 5), bursty.locals());
    assert_eq!(GlobalTrace::expected(1, 1, 5), bursty.global().events());
}

#[test]
fn single_thread_n_steps_n_iterations() {
    let mut builder = GlobalTrace::create_builder(1);

    builder.add_complex_step(|| GlobalTrace::create_step(0));
    builder.add_complex_step(|| GlobalTrace::create_step(1));
    builder.add_complex_step(|| GlobalTrace::create_step(2));
    builder.add_complex_step(|| GlobalTrace::create_step(3));
    builder.add_complex_step(|| GlobalTrace::create_step(4));

    let bursty = builder.launch(3);

    assert_eq!(LocalTrace::expected(1, 3, 5), bursty.locals());
    assert_eq!(GlobalTrace::expected(1, 3, 5), bursty.global().events());
}

#[test]
fn n_threads_single_step_single_iteration() {
    let mut builder = GlobalTrace::create_builder(3);

    builder.add_complex_step(|| GlobalTrace::create_step(0));

    let bursty = builder.launch(1);

    assert_eq!(LocalTrace::expected(3, 1, 1), bursty.locals());
    assert_eq!(GlobalTrace::expected(3, 1, 1), bursty.global().events());
}

#[test]
fn n_threads_single_step_n_iterations() {
    let mut builder = GlobalTrace::create_builder(3);

    builder.add_complex_step(|| GlobalTrace::create_step(0));

    let bursty = builder.launch(5);

    assert_eq!(LocalTrace::expected(3, 5, 1), bursty.locals());
    assert_eq!(GlobalTrace::expected(3, 5, 1), bursty.global().events());
}

#[test]
fn n_threads_n_steps_single_iteration() {
    let mut builder = GlobalTrace::create_builder(3);

    builder.add_complex_step(|| GlobalTrace::create_step(0));
    builder.add_complex_step(|| GlobalTrace::create_step(1));
    builder.add_complex_step(|| GlobalTrace::create_step(2));
    builder.add_complex_step(|| GlobalTrace::create_step(3));
    builder.add_complex_step(|| GlobalTrace::create_step(4));

    let bursty = builder.launch(1);

    assert_eq!(LocalTrace::expected(3, 1, 5), bursty.locals());
    assert_eq!(GlobalTrace::expected(3, 1, 5), bursty.global().events());
}

#[test]
fn n_threads_n_steps_n_iterations() {
    let mut builder = GlobalTrace::create_builder(3);

    builder.add_complex_step(|| GlobalTrace::create_step(0));
    builder.add_complex_step(|| GlobalTrace::create_step(1));
    builder.add_complex_step(|| GlobalTrace::create_step(2));
    builder.add_complex_step(|| GlobalTrace::create_step(3));
    builder.add_complex_step(|| GlobalTrace::create_step(4));
    builder.add_complex_step(|| GlobalTrace::create_step(5));
    builder.add_complex_step(|| GlobalTrace::create_step(6));

    let bursty = builder.launch(5);

    assert_eq!(LocalTrace::expected(3, 5, 7), bursty.locals());
    assert_eq!(GlobalTrace::expected(3, 5, 7), bursty.global().events());
}

} // mod tests
