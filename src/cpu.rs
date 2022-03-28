use custos::{cpu::{CPUCache, InternCPU}, Matrix, number::Number};


pub fn scalar_apply<T: Number, F: Fn(&mut T, T, T)>(device: &InternCPU, lhs: Matrix<T>, scalar: T, f: F) -> Matrix<T> {
    let mut y = CPUCache::get::<T>(device.clone(), lhs.dims());
    let lhs = lhs.as_cpu_slice();
    for (idx, value) in y.as_cpu_slice_mut().iter_mut().enumerate() {
        f(value, lhs[idx], scalar)
    }
    y
}