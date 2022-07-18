mod assign_to_lhs;
mod correlate;
mod ew;

pub use assign_to_lhs::*;
pub use correlate::*;
pub use ew::*;

use custos::{
    cpu::{CPUCache, CPU},
    number::Number,
};

use crate::Matrix;

pub fn cached<T: Default + Copy>(device: &CPU, dims: (usize, usize)) -> Matrix<T> {
    (CPUCache::get::<T>(device, dims.0 * dims.1), dims).into()
}

pub fn scalar_apply<T: Number, F: Fn(&mut T, T, T)>(
    device: &CPU,
    lhs: &Matrix<T>,
    scalar: T,
    f: F,
) -> Matrix<T> {
    let mut y = cached(device, lhs.dims());
    
    for (idx, value) in y.iter_mut().enumerate() {
        f(value, lhs[idx], scalar)
    }
    y
}

pub fn row_op_slice_mut<T: Copy, F: Fn(&mut T, T, T)>(lhs: &[T], lhs_rows: usize, lhs_cols: usize, rhs: &[T], out: &mut [T], f: F) {
    for i in 0..lhs_rows {
        let index = i * lhs_cols;
        let x = &lhs[index..index + lhs_cols];

        for (idx, value) in rhs.iter().enumerate() {
            f(&mut out[index + idx], x[idx], *value);
        }
    }
}

pub fn row_op_slice_lhs<T: Copy, F: Fn(&mut T, T)>(lhs: &mut [T], lhs_rows: usize, lhs_cols: usize, rhs: &[T], f: F) {
    for i in 0..lhs_rows {
        let index = i * lhs_cols;

        for (idx, value) in rhs.iter().enumerate() {
            f(&mut lhs[index + idx], *value);
        }
    }
}

pub fn row_op<T: Number, F: Fn(&mut T, T, T)>(
    device: &CPU,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    assert!(rhs.rows() == 1 && rhs.cols() == lhs.cols());

    let mut out = cached(device, lhs.dims());
    row_op_slice_mut(lhs, lhs.rows(), lhs.cols(), rhs, &mut out, f);
    out
}

pub fn col_op<T: Number, F: Fn(&mut T, T, T)>(
    device: &CPU,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    let mut out = cached(device, lhs.dims());


    // TODO: refactor to function
    //rows
    let mut i = 0;
    for (idx, rdata_value) in rhs.iter().enumerate().take(lhs.rows()) {
        let index = idx * lhs.cols();
        let row = &lhs[index..index + lhs.cols()];
        for data in row {
            f(&mut out[i], *data, *rdata_value);
            i += 1;
        }
    }
    out
}

pub fn each_op<T: Copy + Default, F: Fn(T) -> T>(device: &CPU, x: &Matrix<T>, f: F) -> Matrix<T> {
    let mut out = CPUCache::get::<T>(device, x.size());

    for (idx, value) in out.iter_mut().enumerate() {
        *value = f(x[idx]);
    }
    (out, x.dims()).into()
}
