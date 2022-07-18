pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
mod matrix;
#[cfg(feature = "opencl")]
pub mod opencl;
mod ops;
mod syntax;
pub use matrix::Matrix;

pub use cpu::*;
#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "opencl")]
pub use opencl::*;
pub use ops::*;

pub trait Mat<T> {
    fn as_mat(&self) -> &Matrix<T>;
    fn as_mat_mut(&mut self) -> &mut Matrix<T>;
}

impl<T> Mat<T> for Matrix<T> {
    fn as_mat(&self) -> &Matrix<T> {
        self
    }

    fn as_mat_mut(&mut self) -> &mut Matrix<T> {
        self
    }
}

impl<T> Mat<T> for &Matrix<T> {
    fn as_mat(&self) -> &Matrix<T> {
        self
    }

    fn as_mat_mut(&mut self) -> &mut Matrix<T> {
        unimplemented!()
    }
}

impl<T> Mat<T> for &mut Matrix<T> {
    fn as_mat(&self) -> &Matrix<T> {
        self
    }

    fn as_mat_mut(&mut self) -> &mut Matrix<T> {
        self
    }
}