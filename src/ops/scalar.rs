use crate::{cpu::scalar_apply, Matrix};
use custos::{impl_stack, number::Number, CDatatype, Device, MainMemory, Shape};

#[cfg(feature = "cpu")]
use custos::CPU;

#[cfg(feature = "stack")]
use custos::Stack;

#[cfg(feature = "opencl")]
use crate::opencl::cl_scalar_op_mat;
#[cfg(feature = "opencl")]
use custos::OpenCL;

#[cfg(feature = "cuda")]
use crate::cuda::cu_scalar_op;
#[cfg(feature = "cuda")]
use custos::CUDA;

impl<'a, T, D, S> Matrix<'a, T, D, S>
where
    D: AdditionalOps<T, S>,
    S: Shape,
{
    #[inline]
    pub fn adds(&self, rhs: T) -> Self {
        self.device().adds(self, rhs)
    }

    #[inline]
    pub fn subs(&self, rhs: T) -> Self {
        self.device().subs(self, rhs)
    }

    #[inline]
    pub fn muls(&self, rhs: T) -> Self {
        self.device().muls(self, rhs)
    }

    #[inline]
    pub fn divs(&self, rhs: T) -> Self {
        self.device().divs(self, rhs)
    }

    #[inline]
    pub fn rems(&self, rhs: T) -> Self {
        self.device().rems(self, rhs)
    }
}

pub trait AdditionalOps<T, S: Shape = (), D: Device = Self>: Device {
    fn adds(&self, lhs: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S>;
    fn subs(&self, lhs: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S>;
    fn muls(&self, lhs: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S>;
    fn divs(&self, lhs: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S>;
    fn rems(&self, lhs: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S>;
}

#[cfg(feature = "cuda")]
impl<T: CDatatype + Number> AdditionalOps<T> for CUDA {
    #[inline]
    fn adds(&self, lhs: &Matrix<T, CUDA>, rhs: T) -> Matrix<T, CUDA> {
        (cu_scalar_op(self, lhs, rhs, "+").unwrap(), lhs.dims()).into()
    }

    #[inline]
    fn muls(&self, lhs: &Matrix<T, CUDA>, rhs: T) -> Matrix<T, CUDA> {
        (cu_scalar_op(self, lhs, rhs, "*").unwrap(), lhs.dims()).into()
    }

    #[inline]
    fn divs(&self, lhs: &Matrix<T, CUDA>, rhs: T) -> Matrix<T, CUDA> {
        (cu_scalar_op(self, lhs, rhs, "/").unwrap(), lhs.dims()).into()
    }

    #[inline]
    fn subs(&self, lhs: &Matrix<T, Self>, rhs: T) -> Matrix<T, Self, ()> {
        (cu_scalar_op(self, lhs, rhs, "-").unwrap(), lhs.dims()).into()
    }

    #[inline]
    fn rems(&self, lhs: &Matrix<T, Self, ()>, rhs: T) -> Matrix<T, Self, ()> {
        (cu_scalar_op(self, lhs, rhs, "%").unwrap(), lhs.dims()).into()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype + Number> AdditionalOps<T> for OpenCL {
    #[inline]
    fn adds(&self, lhs: &Matrix<T, Self>, rhs: T) -> Matrix<T, Self> {
        cl_scalar_op_mat(self, lhs, rhs, "+").unwrap()
    }

    #[inline]
    fn subs(&self, lhs: &Matrix<T, Self, ()>, rhs: T) -> Matrix<T, Self, ()> {
        cl_scalar_op_mat(self, lhs, rhs, "-").unwrap()
    }

    #[inline]
    fn muls(&self, lhs: &Matrix<T, Self>, rhs: T) -> Matrix<T, Self> {
        cl_scalar_op_mat(self, lhs, rhs, "*").unwrap()
    }

    #[inline]
    fn divs(&self, lhs: &Matrix<T, Self>, rhs: T) -> Matrix<T, Self> {
        cl_scalar_op_mat(self, lhs, rhs, "/").unwrap()
    }

    #[inline]
    fn rems(&self, lhs: &Matrix<T, Self>, rhs: T) -> Matrix<T, Self> {
        cl_scalar_op_mat(self, lhs, rhs, "%").unwrap()
    }
}

#[impl_stack]
impl<T: Number, D: MainMemory, S: Shape> AdditionalOps<T, S, D> for CPU {
    #[inline]
    fn adds(&self, lhs: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a + b)
    }

    #[inline]
    fn subs(&self, lhs: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a - b)
    }

    #[inline]
    fn muls(&self, lhs: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a * b)
    }

    #[inline]
    fn divs(&self, lhs: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a / b)
    }

    #[inline]
    fn rems(&self, lhs: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a % b)
    }
}
