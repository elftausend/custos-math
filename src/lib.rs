mod cpu;
mod opencl;
mod ops;
mod syntax;
#[cfg(feature="cuda")]
mod cuda;

pub use cpu::*;
use custos::Matrix;
pub use ops::*;

pub trait Mat<T> {
    fn as_mat(&self) -> &Matrix<T>;
}

impl<T> Mat<T> for Matrix<T> {
    fn as_mat(&self) -> &Matrix<T> {
        self
    }
}

impl<T> Mat<T> for &Matrix<T> {
    fn as_mat(&self) -> &Matrix<T> {
        self
    }
}
