use custos::CacheReturn;
use custos_math::{
    custos::{Ident, CPU},
    Matrix,
};

fn main() {
    let device = CPU::new();

    let a = Matrix::<i16>::new(&device, (100, 100));
    let b = Matrix::<i16>::new(&device, (100, 100));

    let out = a + b;
    let ptr = {
        let cache = device.cache();
        let mut node = Ident::new(100 * 100);
        node.idx = 0;
        cache.nodes.get(&node).unwrap().ptr
    };

    assert!(ptr == out.as_buf().ptr.ptr as *mut u8);
}
