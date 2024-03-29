#![cfg_attr(feature = "no-std", no_std)]

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
mod matrix;
#[cfg(feature = "opencl")]
pub mod opencl;
mod ops;
mod syntax;
pub use matrix::Matrix;

pub mod raw_ops;
pub mod raw_prelude;

pub use raw_ops::*;

pub use cpu::*;
#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "opencl")]
pub use opencl::*;
pub use ops::*;

pub mod matrix_multiply;

pub mod custos {
    pub use custos::*;
}
