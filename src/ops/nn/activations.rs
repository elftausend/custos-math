use crate::{opencl::cl_str_op, cu_str_op};
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
        cl_str_op(self, x, "1.0 / (1.0 + exp(-x))").unwrap()
    }

    fn tanh(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_str_op(self, x, "tanh(x)").unwrap()
    }

    fn tanh_grad(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_str_op(self, x, "1.0 - pow(tanh(x), 2)").unwrap()
    }

    fn relu(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_str_op(self, x, "x * (x >= 0)").unwrap()
    }

    fn relu_grad(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_str_op(self, x, "(x >= 0)").unwrap()
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

impl<T: CDatatype> ActivationOps<T> for CudaDevice {
    fn sigmoid(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "1.0 / (1.0 + exp(-x))").unwrap();
        (out, x.dims()).into()
    }

    fn tanh(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "tanh(x)").unwrap();
        (out, x.dims()).into()
    }

    fn tanh_grad(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "1.0 - pow(tanh(x), 2)").unwrap();
        (out, x.dims()).into()
    }

    fn relu(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "x * (x >= 0)").unwrap();
        (out, x.dims()).into()
    }

    fn relu_grad(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "(x >= 0)").unwrap();
        (out, x.dims()).into()
    }
}
