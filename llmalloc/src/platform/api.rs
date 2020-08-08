//! API of OS required services.

pub use llmalloc_core::Configuration;

/// Abstraction over OS services.
pub(crate) trait Platform : llmalloc_core::Platform + Send + Sync {
    /// Returns the current NUMA node on which the thread is running.
    ///
    /// As the thread may migrate to another node at the scheduler's whim, the actual result has no impact on
    /// correctness. It does, however, impact performance: it is better for a node's thread to access memory
    /// stored in the node's memory banks, rather than another node.
    fn current_node(&self) -> NumaNodeIndex;
}

/// Abstraction over thread-local storage.
pub(crate) trait ThreadLocal<T> {
    /// Returns a pointer to the thread-local value associated to this instance.
    ///
    /// May return a null pointer if no prior value was set, or it was already destructed.
    fn get(&self) -> *mut T;

    /// Sets the pointer to the thread-local value associated to this instance.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the value is not already set.
    fn set(&self, value: *mut T);
}

/// Index of a NUMA node.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub(crate) struct NumaNodeIndex(u32);

impl NumaNodeIndex {
    /// Creates a NumaNodeIndex.
    pub(crate) fn new(value: u32) -> Self { Self(value) }

    /// Retrieves the index.
    pub(crate) fn value(&self) -> u32 { self.0 }
}
