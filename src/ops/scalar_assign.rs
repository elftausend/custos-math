use crate::{assign_to_lhs_scalar, Matrix};
use custos::{CPU, CDatatype};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

#[cfg(feature = "opencl")]
use custos::OpenCL;
#[cfg(feature = "opencl")]
use crate::cl_assign_scalar;

#[cfg(feature = "cuda")]
use custos::CudaDevice;
#[cfg(feature = "cuda")]
use crate::cu_assign_scalar;

pub trait ScalarAssign<T> {
    fn adds_assign(&self, lhs: &mut Matrix<T>, rhs: T);
    fn muls_assign(&self, lhs: &mut Matrix<T>, rhs: T);
    fn divs_assign(&self, lhs: &mut Matrix<T>, rhs: T);
    fn subs_assign(&self, lhs: &mut Matrix<T>, rhs: T);
}

impl<T> ScalarAssign<T> for CPU
where
    T: Copy + AddAssign + MulAssign + DivAssign + SubAssign,
{
    fn adds_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        assign_to_lhs_scalar(lhs, rhs, |x, y| *x += y);
    }

    fn muls_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        assign_to_lhs_scalar(lhs, rhs, |x, y| *x *= y);
    }

    fn divs_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        assign_to_lhs_scalar(lhs, rhs, |x, y| *x /= y);
    }

    fn subs_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        assign_to_lhs_scalar(lhs, rhs, |x, y| *x -= y);
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> ScalarAssign<T> for OpenCL {
    fn adds_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        cl_assign_scalar(self, lhs, rhs, "+").unwrap();
    }

    fn muls_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        cl_assign_scalar(self, lhs, rhs, "*").unwrap();
    }

    fn divs_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        cl_assign_scalar(self, lhs, rhs, "/").unwrap();
    }

    fn subs_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        cl_assign_scalar(self, lhs, rhs, "-").unwrap();
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> ScalarAssign<T> for CudaDevice {
    fn adds_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        cu_assign_scalar(self, lhs, rhs, "+").unwrap();
    }

    fn muls_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        cu_assign_scalar(self, lhs, rhs, "*").unwrap();
    }

    fn divs_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        cu_assign_scalar(self, lhs, rhs, "/").unwrap();
    }

    fn subs_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        cu_assign_scalar(self, lhs, rhs, "-").unwrap();
    }
}

impl<'a, T: CDatatype> AddAssign<T> for Matrix<'a, T> {
    fn add_assign(&mut self, rhs: T) {
        todo!()
        //self.device().adds_assign(self, rhs);
    }
}

impl<T: CDatatype> MulAssign<T> for Matrix<'_, T> {
    fn mul_assign(&mut self, rhs: T) {
        todo!()
        //self.device().muls_assign(self, rhs);
    }
}

impl<T: CDatatype> DivAssign<T> for Matrix<'_, T> {
    fn div_assign(&mut self, rhs: T) {
        todo!()
        //self.device().divs_assign(self, rhs);
    }
}
