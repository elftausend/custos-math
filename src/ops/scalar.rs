use custos::{cpu::InternCPU, get_device, opencl::InternCLDevice, CDatatype, Matrix, number::Number};

use crate::{cpu::scalar_apply, opencl::scalar_op};

pub trait Additional<T> {
    fn adds(&self, rhs: T) -> Matrix<T>;
    fn muls(&self, rhs: T) -> Matrix<T>;
    fn divs(&self, rhs: T) -> Matrix<T>;
}

impl<T: CDatatype> Additional<T> for Matrix<T> {
    fn adds(&self, rhs: T) -> Matrix<T> {
        let device = get_device!(AdditionalOps, T).unwrap();
        device.adds(self, rhs)
    }

    fn muls(&self, rhs: T) -> Matrix<T> {
        let device = get_device!(AdditionalOps, T).unwrap();
        device.muls(self, rhs)
    }

    fn divs(&self, rhs: T) -> Matrix<T> {
        let device = get_device!(AdditionalOps, T).unwrap();
        device.divs(self, rhs)
    }
}

pub trait AdditionalOps<T> {
    fn adds(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T>;
    fn muls(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T>;
    fn divs(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T>;
}

impl<T: CDatatype> AdditionalOps<T> for InternCLDevice {
    fn adds(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        scalar_op(self.clone(), lhs, rhs, "+").unwrap()
    }

    fn muls(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        scalar_op(self.clone(), lhs, rhs, "*").unwrap()
    }

    fn divs(&self, lhs: &Matrix<T>, rhs: T) -> Matrix<T> {
        scalar_op(self.clone(), lhs, rhs, "/").unwrap()
    }
}

impl <T: Number>AdditionalOps<T> for InternCPU {
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
