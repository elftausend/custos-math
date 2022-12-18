use crate::{cpu::scalar_apply, Matrix};
use custos::{cpu::CPU, number::Number, CDatatype, Device, MainMemory};

#[cfg(feature = "opencl")]
use crate::opencl::cl_scalar_op_mat;
#[cfg(feature = "opencl")]
use custos::OpenCL;

#[cfg(feature = "cuda")]
use crate::cuda::cu_scalar_op;
#[cfg(feature = "cuda")]
use custos::CudaDevice;

impl<'a, T: CDatatype, D: Device> Matrix<'a, T, D>
where
    D: AdditionalOps<T, D>,
{
    pub fn adds(&self, rhs: T) -> Matrix<'a, T, D> {
        self.device().adds(self, rhs)
    }

    pub fn muls(&self, rhs: T) -> Matrix<'a, T, D> {
        self.device().muls(self, rhs)
    }

    pub fn divs(&self, rhs: T) -> Matrix<'a, T, D> {
        self.device().divs(self, rhs)
    }
}

pub trait AdditionalOps<T, D: Device>: Device {
    fn adds(&self, lhs: &Matrix<T, D>, rhs: T) -> Matrix<T, Self>;
    fn muls(&self, lhs: &Matrix<T, D>, rhs: T) -> Matrix<T, Self>;
    fn divs(&self, lhs: &Matrix<T, D>, rhs: T) -> Matrix<T, Self>;
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
impl<T: CDatatype> AdditionalOps<T> for OpenCL {
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

impl<T: Number, D: MainMemory> AdditionalOps<T, D> for CPU {
    fn adds(&self, lhs: &Matrix<T, D>, rhs: T) -> Matrix<T> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a + b)
    }

    fn muls(&self, lhs: &Matrix<T, D>, rhs: T) -> Matrix<T> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a * b)
    }

    fn divs(&self, lhs: &Matrix<T, D>, rhs: T) -> Matrix<T> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a / b)
    }
}
