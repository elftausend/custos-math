mod opencl;
mod ops;
mod cpu;
mod syntax;

use custos::Matrix;
pub use ops::*;
pub use cpu::*;

pub trait Mat<T> {
    fn as_slice(&self) -> Matrix<T>;
}