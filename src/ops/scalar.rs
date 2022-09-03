use crate::{cpu::scalar_apply, Matrix};
use custos::{cpu::CPU, get_device, number::Number, CDatatype};

#[cfg(feature = "opencl")]
use crate::opencl::cl_scalar_op_mat;
#[cfg(feature = "opencl")]
use custos::CLDevice;

#[cfg(feature = "cuda")]
use crate::cuda::cu_scalar_op;
#[cfg(feature = "cuda")]
use custos::CudaDevice;

impl<'a, T: CDatatype> Matrix<'a, T> {
    pub fn adds(&self, rhs: T) -> Matrix<'a, T> {
        get_device!(self.device(), AdditionalOps<T>).adds(self, rhs)
    }

    pub fn muls(&self, rhs: T) -> Matrix<'a, T> {
        get_device!(self.device(), AdditionalOps<T>).muls(self, rhs)
    }

    pub fn divs(&self, rhs: T) -> Matrix<'a, T> {
        get_device!(self.device(), AdditionalOps<T>).divs(self, rhs)
    }
}

pub trait AdditionalOps<T> {
    fn adds(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T>;
    fn muls(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T>;
    fn divs(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T>;
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> AdditionalOps<T> for CudaDevice {
    fn adds(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        (cu_scalar_op(self, lhs, rhs, "+").unwrap(), lhs.dims()).into()
    }

    fn muls(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        (cu_scalar_op(self, lhs, rhs, "*").unwrap(), lhs.dims()).into()
    }

    fn divs(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        (cu_scalar_op(self, lhs, rhs, "/").unwrap(), lhs.dims()).into()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> AdditionalOps<T> for CLDevice {
    fn adds(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        cl_scalar_op_mat(self, lhs, rhs, "+").unwrap()
    }

    fn muls(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        cl_scalar_op_mat(self, lhs, rhs, "*").unwrap()
    }

    fn divs(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        cl_scalar_op_mat(self, lhs, rhs, "/").unwrap()
    }
}

impl<T: Number> AdditionalOps<T> for CPU {
    fn adds(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a + b)
    }

    fn muls(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a * b)
    }

    fn divs(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a / b)
    }
}
