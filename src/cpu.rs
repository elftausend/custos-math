use custos::{cpu::{CPUCache, InternCPU}, Matrix, number::Number};

pub fn scalar_apply<T: Number, F: Fn(&mut T, T, T)>(device: &InternCPU, lhs: Matrix<T>, scalar: T, f: F) -> Matrix<T> {
    let mut y = CPUCache::get::<T>(device.clone(), lhs.dims());
    let lhs = lhs.as_cpu_slice();
    for (idx, value) in y.as_cpu_slice_mut().iter_mut().enumerate() {
        f(value, lhs[idx], scalar)
    }
    y
}

pub fn row_op<T: Number, F: Fn(&mut T, T, T)>(device: &InternCPU, lhs: Matrix<T>, rhs: Matrix<T>, f: F) -> Matrix<T> {
    assert!(rhs.dims().0 == 1);
    
    let mut y = CPUCache::get::<T>(device.clone(), lhs.dims());
    let lhs_data = lhs.as_cpu_slice();
    let rhs_data = rhs.as_cpu_slice();

    //rows
    for i in 0..lhs.dims().0 {

        let index = i*lhs.dims().1;
        let x = &lhs_data[index..index+lhs.dims().1];    

        for (idx, value) in rhs_data.iter().enumerate() {
            f(&mut y.as_cpu_slice_mut()[index + idx], x[idx], *value);
        }
    }
    y
}