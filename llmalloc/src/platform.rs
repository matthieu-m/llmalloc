//! Abstraction over OS differences.

mod api;

pub(crate) use api::{NumaNodeIndex, Configuration, Platform, ThreadLocal};

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "linux")]
pub(crate) use linux::{LLConfiguration, LLPlatform, LLThreadLocal};
