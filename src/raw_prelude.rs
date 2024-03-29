pub use crate::{assign_to_lhs, assign_to_lhs_scalar, scalar_apply, slice_transpose};

#[cfg(feature = "fastrand")]
pub use crate::rand_slice;

#[cfg(feature = "opencl")]
pub use crate::{
    cl_diagflat, cl_gemm, cl_scalar_op, cl_str_op, cl_tew, cl_tew_self, cl_transpose, cl_write,
};
