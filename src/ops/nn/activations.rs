use crate::opencl::str_op;
use custos::{
    get_device,
    libs::{
        cpu::{each_op, CPU},
        opencl::cl_device::CLDevice,
    },
    number::Float,
    CDatatype, Matrix, CudaDevice,
};

pub trait Activations<T> {
    fn tanh(&self) -> Matrix<T>;
    fn tanh_grad(&self) -> Matrix<T>;
    fn relu(&self) -> Matrix<T>;
    fn relu_grad(&self) -> Matrix<T>;
}

impl<T: CDatatype + Float> Activations<T> for Matrix<T> {
    fn tanh(&self) -> Matrix<T> {
        let device = get_device!(ActivationOps, T).unwrap();
        device.tanh(self)
    }

    fn tanh_grad(&self) -> Matrix<T> {
        let device = get_device!(ActivationOps, T).unwrap();
        device.tanh_grad(self)
    }

    fn relu(&self) -> Matrix<T> {
        let device = get_device!(ActivationOps, T).unwrap();
        device.relu(self)
    }

    fn relu_grad(&self) -> Matrix<T> {
        let device = get_device!(ActivationOps, T).unwrap();
        device.relu_grad(self)
    }
}

pub trait ActivationOps<T> {
    fn sigmoid(&self, x: &Matrix<T>) -> Matrix<T>;
    fn tanh(&self, x: &Matrix<T>) -> Matrix<T>;
    fn tanh_grad(&self, x: &Matrix<T>) -> Matrix<T>;
    fn relu(&self, x: &Matrix<T>) -> Matrix<T>;
    fn relu_grad(&self, x: &Matrix<T>) -> Matrix<T>;
}

impl<T: CDatatype + Float> ActivationOps<T> for CLDevice {
    fn sigmoid(&self, x: &Matrix<T>) -> Matrix<T> {
        str_op(self, x, "1.0 / (1.0 + exp(-I))").unwrap()
    }

    fn tanh(&self, x: &Matrix<T>) -> Matrix<T> {
        str_op(self, x, "tanh(I)").unwrap()
    }

    fn tanh_grad(&self, x: &Matrix<T>) -> Matrix<T> {
        str_op(self, x, "1.0 - pow(tanh(I), 2)").unwrap()
    }

    fn relu(&self, x: &Matrix<T>) -> Matrix<T> {
        str_op(self, x, "I * (I >= 0)").unwrap()
    }

    fn relu_grad(&self, x: &Matrix<T>) -> Matrix<T> {
        str_op(self, x, "(I >= 0)").unwrap()
    }
}

impl<T: Float> ActivationOps<T> for CPU {
    fn sigmoid(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| T::one() / (T::one() + x.negate().exp()))
    }

    fn tanh(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.tanh())
    }

    fn tanh_grad(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| T::one() - x.tanh().powi(2))
    }

    fn relu(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| T::from_usize((x >= T::zero()) as usize) * x)
    }

    fn relu_grad(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| T::from_usize((x >= T::zero()) as usize))
    }
}

impl<T> ActivationOps<T> for CudaDevice {
    fn sigmoid(&self, x: &Matrix<T>) -> Matrix<T> {
        todo!()
    }

    fn tanh(&self, x: &Matrix<T>) -> Matrix<T> {
        todo!()
    }

    fn tanh_grad(&self, x: &Matrix<T>) -> Matrix<T> {
        todo!()
    }

    fn relu(&self, x: &Matrix<T>) -> Matrix<T> {
        todo!()
    }

    fn relu_grad(&self, x: &Matrix<T>) -> Matrix<T> {
        todo!()
    }
}
