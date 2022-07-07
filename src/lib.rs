mod cpu;
#[cfg(feature="opencl")]
mod opencl;
mod ops;
mod syntax;

#[cfg(feature="cuda")]
mod cuda;

pub use cpu::*;
#[cfg(feature="cuda")]
pub use cuda::*;
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
