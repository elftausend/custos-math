use custos::{
    cpu::{CPUCache, InternCPU},
    number::Number,
    Matrix,
};

pub fn cached<T: Default + Copy>(device: &InternCPU, dims: (usize, usize)) -> Matrix<T> {
    CPUCache::get::<T>(device.clone(), dims)
}

pub fn scalar_apply<T: Number, F: Fn(&mut T, T, T)>(
    device: &InternCPU,
    lhs: &Matrix<T>,
    scalar: T,
    f: F,
) -> Matrix<T> {
    let mut y = CPUCache::get::<T>(device.clone(), lhs.dims());
    let lhs = lhs.as_cpu_slice();
    for (idx, value) in y.as_cpu_slice_mut().iter_mut().enumerate() {
        f(value, lhs[idx], scalar)
    }
    y
}

pub fn row_op<T: Number, F: Fn(&mut T, T, T)>(
    device: &InternCPU,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    assert!(rhs.rows() == 1 && rhs.cols() == lhs.cols());

    let mut y = CPUCache::get::<T>(device.clone(), lhs.dims());
    let lhs_data = lhs.as_cpu_slice();
    let rhs_data = rhs.as_cpu_slice();

    //rows
    for i in 0..lhs.rows() {
        let index = i * lhs.dims().1;
        let x = &lhs_data[index..index + lhs.dims().1];

        for (idx, value) in rhs_data.iter().enumerate() {
            f(&mut y.as_cpu_slice_mut()[index + idx], x[idx], *value);
        }
    }
    y
}

pub fn col_op<T: Number, F: Fn(&mut T, T, T)>(
    device: &InternCPU,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    let mut y = CPUCache::get::<T>(device.clone(), lhs.dims());

    let lhs_data = lhs.as_cpu_slice();
    let rhs_data = rhs.as_cpu_slice();
    let y_slice = y.as_cpu_slice_mut();

    //rows
    let mut i = 0;
    for (idx, rdata_value) in rhs_data.iter().enumerate().take(lhs.rows()) {
        let index = idx * lhs.cols();
        let row = &lhs_data[index..index + lhs.cols()];
        for data in row {
            f(&mut y_slice[i], *data, *rdata_value);
            i += 1;
        }
    }
    y
}
