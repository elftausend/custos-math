pub use crate::{rand_slice, scalar_apply, slice_transpose, assign_to_lhs, assign_to_lhs_scalar};

#[cfg(feature="opencl")]
pub use crate::{cl_tew_self, cl_transpose, cl_write, cl_diagflat, cl_gemm, cl_scalar_op, cl_str_op, cl_tew};