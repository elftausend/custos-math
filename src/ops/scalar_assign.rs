use crate::{assign_to_lhs_scalar, Matrix};
use core::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use custos::{impl_stack, CDatatype, Device, MainMemory, Shape, CPU};

#[cfg(feature = "stack")]
use custos::Stack;

#[cfg(feature = "opencl")]
use crate::cl_assign_scalar;
#[cfg(feature = "opencl")]
use custos::OpenCL;

#[cfg(feature = "cuda")]
use crate::cu_assign_scalar;
#[cfg(feature = "cuda")]
use custos::CUDA;

impl<'a, T, S, D> AddAssign<T> for Matrix<'a, T, D, S>
where
    S: Shape,
    D: ScalarAssign<T, S>,
{
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.device().adds_assign(self, rhs)
    }
}

impl<T, S, D> MulAssign<T> for Matrix<'_, T, D, S>
where
    S: Shape,
    D: ScalarAssign<T, S>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.device().muls_assign(self, rhs);
    }
}

impl<T, S, D> DivAssign<T> for Matrix<'_, T, D, S>
where
    S: Shape,
    D: ScalarAssign<T, S>,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.device().divs_assign(self, rhs);
    }
}

pub trait ScalarAssign<T, S: Shape = (), D: Device = Self>: Device {
    fn adds_assign(&self, lhs: &mut Matrix<T, D, S>, rhs: T);
    fn muls_assign(&self, lhs: &mut Matrix<T, D, S>, rhs: T);
    fn divs_assign(&self, lhs: &mut Matrix<T, D, S>, rhs: T);
    fn subs_assign(&self, lhs: &mut Matrix<T, D, S>, rhs: T);
}

#[impl_stack]
impl<T, D: MainMemory, S: Shape> ScalarAssign<T, S, D> for CPU
where
    T: Copy + AddAssign + MulAssign + DivAssign + SubAssign,
{
    #[inline]
    fn adds_assign(&self, lhs: &mut Matrix<T, D, S>, rhs: T) {
        assign_to_lhs_scalar(lhs, rhs, |x, y| *x += y);
    }

    #[inline]
    fn muls_assign(&self, lhs: &mut Matrix<T, D, S>, rhs: T) {
        assign_to_lhs_scalar(lhs, rhs, |x, y| *x *= y);
    }

    #[inline]
    fn divs_assign(&self, lhs: &mut Matrix<T, D, S>, rhs: T) {
        assign_to_lhs_scalar(lhs, rhs, |x, y| *x /= y);
    }

    #[inline]
    fn subs_assign(&self, lhs: &mut Matrix<T, D, S>, rhs: T) {
        assign_to_lhs_scalar(lhs, rhs, |x, y| *x -= y);
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> ScalarAssign<T> for OpenCL {
    #[inline]
    fn adds_assign(&self, lhs: &mut Matrix<T, Self>, rhs: T) {
        cl_assign_scalar(self, lhs, rhs, "+").unwrap();
    }

    #[inline]
    fn muls_assign(&self, lhs: &mut Matrix<T, Self>, rhs: T) {
        cl_assign_scalar(self, lhs, rhs, "*").unwrap();
    }

    fn divs_assign(&self, lhs: &mut Matrix<T, Self>, rhs: T) {
        cl_assign_scalar(self, lhs, rhs, "/").unwrap();
    }

    fn subs_assign(&self, lhs: &mut Matrix<T, Self>, rhs: T) {
        cl_assign_scalar(self, lhs, rhs, "-").unwrap();
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> ScalarAssign<T> for CUDA {
    #[inline]
    fn adds_assign(&self, lhs: &mut Matrix<T, CUDA>, rhs: T) {
        cu_assign_scalar(self, lhs, rhs, "+").unwrap();
    }

    #[inline]
    fn muls_assign(&self, lhs: &mut Matrix<T, CUDA>, rhs: T) {
        cu_assign_scalar(self, lhs, rhs, "*").unwrap();
    }

    #[inline]
    fn divs_assign(&self, lhs: &mut Matrix<T, CUDA>, rhs: T) {
        cu_assign_scalar(self, lhs, rhs, "/").unwrap();
    }

    #[inline]
    fn subs_assign(&self, lhs: &mut Matrix<T, CUDA>, rhs: T) {
        cu_assign_scalar(self, lhs, rhs, "-").unwrap();
    }
}
