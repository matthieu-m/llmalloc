//! The API of llmalloc-core.

mod configuration;
mod description;
mod domain;
mod platform;
mod socket;
mod thread;

pub use configuration::{Configuration, Properties};
pub use description::{AllocationSize, Category, ClassSize, Layout, PowerOf2};
pub use domain::DomainHandle;
pub use platform::Platform;
pub use socket::{AtomicSocketHandle, SocketHandle};
pub use thread::ThreadHandle;
