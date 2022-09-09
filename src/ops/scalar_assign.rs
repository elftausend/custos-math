use crate::{assign_to_lhs_scalar, Matrix};
use custos::{get_device, number::Number, CPU};
use std::ops::{AddAssign, DivAssign, MulAssign};

#[cfg(feature = "opencl")]
use custos::CLDevice;

pub trait ScalarAssign<T> {
    fn adds_assign(&self, lhs: &mut Matrix<T>, rhs: T);
    fn muls_assign(&self, lhs: &mut Matrix<T>, rhs: T);
    fn divs_assign(&self, lhs: &mut Matrix<T>, rhs: T);
}

impl<T> ScalarAssign<T> for CPU
where
    T: Copy + AddAssign + MulAssign + DivAssign,
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
}

#[cfg(feature = "opencl")]
impl<T> ScalarAssign<T> for CLDevice {
    fn adds_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        todo!()
    }

    fn muls_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        todo!()
    }

    fn divs_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        todo!()
    }
}

#[cfg(feature = "cuda")]
impl<T> ScalarAssign<T> for custos::CudaDevice {
    fn adds_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        todo!()
    }

    fn muls_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        todo!()
    }

    fn divs_assign(&self, lhs: &mut Matrix<T>, rhs: T) {
        todo!()
    }
}

impl<T: Number> AddAssign<T> for Matrix<'_, T> {
    fn add_assign(&mut self, rhs: T) {
        get_device!(self.device(), ScalarAssign<T>).adds_assign(self, rhs);
    }
}

impl<T: Number> MulAssign<T> for Matrix<'_, T> {
    fn mul_assign(&mut self, rhs: T) {
        get_device!(self.device(), ScalarAssign<T>).muls_assign(self, rhs);
    }
}

impl<T: Number> DivAssign<T> for Matrix<'_, T> {
    fn div_assign(&mut self, rhs: T) {
        get_device!(self.device(), ScalarAssign<T>).divs_assign(self, rhs);
    }
}
