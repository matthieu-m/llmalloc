use llmalloc::LLAllocator;

#[test]
fn warm_up() {
    let allocator = LLAllocator::new();
    allocator.warm_up().expect("Warmed up!");
}

//  FIXME: use sys crates... properly configured for system libraries.
#[link(name = "c")]
extern "C" {}

#[link(name = "numa")]
extern "C" {}

#[link(name = "pthread")]
extern "C" {}
