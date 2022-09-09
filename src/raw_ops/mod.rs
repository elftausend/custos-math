pub mod cpu;

pub use cpu::*;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::*;


#[cfg(feature = "opencl")]
pub use opencl::*;