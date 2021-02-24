//! Blocks
//!
//! A Block represent a unit of allocation.
//!
//! Whilst allocated, the content of the block is purely in the hands of the user. Whilst deallocated, however, the Block
//! storage is reused to store meta-data used to manage the memory.
//!
//! Specifically, the blocks within a LargePage are maintained into a tail-list structure.
//!
//! Note: Blocks are never _constructed_, instead raw memory is reinterpreted as blocks.

mod atomic_block_foreign_list;
mod atomic_block_foreign_stack;
mod atomic_block_foreign;
mod block_foreign;
mod block_foreign_list;
mod block_local;
mod block_ptr;

pub(crate) use atomic_block_foreign_list::AtomicBlockForeignList;
pub(crate) use atomic_block_foreign_stack::AtomicBlockForeignStack;
pub(crate) use atomic_block_foreign::AtomicBlockForeign;
pub(crate) use block_foreign::BlockForeign;
pub(crate) use block_foreign_list::BlockForeignList;
pub(crate) use block_local::{BlockLocal, BlockLocalStack};
pub(crate) use block_ptr::BlockPtr;

use super::atomic::{AtomicLength, AtomicPtr};

#[cfg(test)]
mod test;

#[cfg(test)]
use test::AlignedArray;
