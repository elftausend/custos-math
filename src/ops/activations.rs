use custos::{Matrix, libs::{opencl::{GenericOCL, cl_device::InternCLDevice}, cpu::{each_op, InternCPU}}, number::Float, get_device};

use crate::opencl::str_op;

pub trait Activations<T> {
    fn relu(&self) -> Matrix<T>;
}

impl <T: GenericOCL+Float>Activations<T> for Matrix<T> {
    fn relu(&self) -> Matrix<T> {
        let device = get_device!(ActivationOps, T).unwrap();
        device.relu(*self)
    }
}

pub trait ActivationOps<T> {
    fn sigmoid(&self, x: Matrix<T>) -> Matrix<T>;
    fn relu(&self, x: Matrix<T>) -> Matrix<T>;
    fn relu_grad(&self, x: Matrix<T>) -> Matrix<T>;
}

impl <T: GenericOCL+Float>ActivationOps<T> for InternCLDevice {
    fn sigmoid(&self, x: Matrix<T>) -> Matrix<T> {
        str_op(self.clone(), x, "1.0 / (1.0 + exp(-I))").unwrap()
    }

    fn relu(&self, x: Matrix<T>) -> Matrix<T> {
        str_op(self.clone(), x, "I * (I >= 0)").unwrap()
    }

    fn relu_grad(&self, x: Matrix<T>) -> Matrix<T> {
        str_op(self.clone(), x, "(I >= 0)").unwrap()
    }
}

impl <T: Float>ActivationOps<T> for InternCPU {
    fn sigmoid(&self, x: Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| T::one() / (T::one() + x.negate().exp()))
    }

    fn relu(&self, x: Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| T::from_usize((x >= T::zero()) as usize) * x)
    }

    fn relu_grad(&self, x: Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| T::from_usize((x >= T::zero()) as usize))
    }
}



