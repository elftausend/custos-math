mod opencl;
mod ops;
mod cpu;
mod syntax;

use custos::Matrix;
pub use ops::*;
pub use cpu::*;

pub trait Mat<T> {
    fn as_mat(&self) -> &Matrix<T>;
}

impl <T>Mat<T> for Matrix<T> {
    fn as_mat(&self) -> &Matrix<T> {
        self
    }
}

impl <T>Mat<T> for &Matrix<T> {
    fn as_mat(&self) -> &Matrix<T> {
        self
    }
}