use custos::{cache::Cache, number::Number, CPU};

use crate::Matrix;

pub fn scalar_apply<'a, T: Number, F: Fn(&mut T, T, T)>(
    device: &'a CPU,
    lhs: &Matrix<T>,
    scalar: T,
    f: F,
) -> Matrix<'a, T> {
    let mut out = Cache::get(device, lhs.len, lhs.node.idx);

    scalar_apply_slice(&mut out, lhs, scalar, f);

    (out, lhs.dims()).into()
}

#[inline]
pub fn scalar_apply_slice<T: Copy, F: Fn(&mut T, T, T)>(out: &mut [T], lhs: &[T], scalar: T, f: F) {
    for (idx, value) in out.iter_mut().enumerate() {
        f(value, lhs[idx], scalar)
    }
}

pub fn row_op_slice_mut<T: Copy, F: Fn(&mut T, T, T)>(
    lhs: &[T],
    lhs_rows: usize,
    lhs_cols: usize,
    rhs: &[T],
    out: &mut [T],
    f: F,
) {
    for i in 0..lhs_rows {
        let index = i * lhs_cols;
        let x = &lhs[index..index + lhs_cols];

        for (idx, value) in rhs.iter().enumerate() {
            f(&mut out[index + idx], x[idx], *value);
        }
    }
}

pub fn row_op_slice_lhs<T: Copy, F: Fn(&mut T, T)>(
    lhs: &mut [T],
    lhs_rows: usize,
    lhs_cols: usize,
    rhs: &[T],
    f: F,
) {
    for i in 0..lhs_rows {
        let index = i * lhs_cols;

        for (idx, value) in rhs.iter().enumerate() {
            f(&mut lhs[index + idx], *value);
        }
    }
}

pub fn row_op<'a, T: Number, F: Fn(&mut T, T, T)>(
    device: &'a CPU,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<'a, T> {
    assert!(rhs.rows() == 1 && rhs.cols() == lhs.cols());

    let mut out = Cache::get(device, lhs.len, [lhs.node.idx, rhs.node.idx]);
    row_op_slice_mut(lhs, lhs.rows(), lhs.cols(), rhs, &mut out, f);
    (out, lhs.dims()).into()
}

pub fn col_op<'a, T: Number, F: Fn(&mut T, T, T)>(
    device: &'a CPU,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<'a, T> {
    let mut out = Cache::get(device, lhs.len, [lhs.node.idx, rhs.node.idx]);

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
    (out, lhs.dims()).into()
}

pub fn each_op<'a, T: Copy + Default, F: Fn(T) -> T>(
    device: &'a CPU,
    x: &Matrix<T>,
    f: F,
) -> Matrix<'a, T> {
    let mut out = Cache::get(device, x.size(), x.node.idx);

    for (idx, value) in out.iter_mut().enumerate() {
        *value = f(x[idx]);
    }
    (out, x.dims()).into()
}
