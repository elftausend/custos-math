pub mod cpu;
#[cfg(feature="opencl")]
pub mod opencl;
#[cfg(feature="cuda")]
pub mod cuda;
mod ops;
mod syntax;
mod matrix;
pub use matrix::Matrix;

pub use cpu::*;
#[cfg(feature="cuda")]
pub use cuda::*;
#[cfg(feature="opencl")]
pub use opencl::*;
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
