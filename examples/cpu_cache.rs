use custos::{cpu::CPU_CACHE, AsDev, Node, CPU};
use custos_math::Matrix;

fn main() {
    let device = CPU::new().select();

    let a = Matrix::<i16>::new(&device, (100, 100));
    let b = Matrix::<i16>::new(&device, (100, 100));

    let out = a + b;
    let info = CPU_CACHE.with(|cache| {
        let cache = cache.borrow();
        let mut node = Node::new(100 * 100);
        node.idx = 0;
        *cache.nodes.get(&node).unwrap()
    });
    assert!(info.0 .0 == out.as_buf().ptr.0 as *mut usize);
}
