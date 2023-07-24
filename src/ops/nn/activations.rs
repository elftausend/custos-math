use crate::{each_op, each_op_slice_mut, Matrix};
use custos::{impl_stack, number::Float, CDatatype, Device, MainMemory, Shape};

#[cfg(feature = "cpu")]
use custos::CPU;

#[cfg(feature = "stack")]
use custos::Stack;

#[cfg(feature = "opencl")]
use crate::{cl_str_op_mut, opencl::cl_str_op_mat};
#[cfg(feature = "opencl")]
use custos::OpenCL;

#[cfg(feature = "cuda")]
use crate::{cu_str_op, cu_str_op_mut};
#[cfg(feature = "cuda")]
use custos::CUDA;

impl<'a, T, D: ActivationOps<T, S>, S: Shape> Matrix<'a, T, D, S> {
    #[inline]
    pub fn tanh(&self) -> Matrix<'a, T, D, S> {
        self.device().tanh(self)
    }

    #[inline]
    pub fn tanh_grad(&self) -> Matrix<'a, T, D, S> {
        self.device().tanh_grad(self)
    }

    #[inline]
    pub fn relu(&self) -> Matrix<'a, T, D, S> {
        self.device().relu(self)
    }

    #[inline]
    pub fn relu_mut(&mut self) {
        self.device().relu_mut(self)
    }

    #[inline]
    pub fn relu_grad(&self) -> Matrix<'a, T, D, S> {
        self.device().relu_grad(self)
    }

    #[inline]
    pub fn relu_grad_mut(&mut self) {
        self.device().relu_grad_mut(self)
    }

    #[inline]
    pub fn sigmoid(&self) -> Matrix<'a, T, D, S> {
        self.device().sigmoid(self)
    }

    #[inline]
    /// uses pre-computed sigmoid activation
    pub fn sigmoid_grad(&self) -> Matrix<'a, T, D, S> {
        self.device().sigmoid_grad(self)
    }
}

pub trait ActivationOps<T, S: Shape = (), D: Device = Self>: Device {
    fn sigmoid(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S>;
    fn sigmoid_grad(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S>;
    fn tanh(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S>;
    fn tanh_grad(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S>;
    fn relu(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S>;
    /// inplace
    fn relu_mut(&self, x: &mut Matrix<T, D, S>);
    fn relu_grad(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S>;
    /// inplace
    fn relu_grad_mut(&self, x: &mut Matrix<T, D, S>);
}

#[cfg(feature = "opencl")]
impl<T: CDatatype + Float> ActivationOps<T> for OpenCL {
    #[inline]
    fn sigmoid(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, "1.0 / (1.0 + exp(-x))").unwrap()
    }

    fn sigmoid_grad(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, "x * (1.0 - x)").unwrap()
    }

    #[inline]
    fn tanh(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, "tanh(x)").unwrap()
    }

    #[inline]
    fn tanh_grad(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, "1.0 - pow(tanh(x), 2)").unwrap()
    }

    #[inline]
    fn relu(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, "x * (x >= 0)").unwrap()
    }

    #[inline]
    fn relu_mut(&self, x: &mut Matrix<T, Self, ()>) {
        cl_str_op_mut(self, x, "x * (x >= 0)").unwrap();
    }

    #[inline]
    fn relu_grad(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, "(x >= 0)").unwrap()
    }

    #[inline]
    fn relu_grad_mut(&self, x: &mut Matrix<T, Self, ()>) {
        cl_str_op_mut(self, x, "(x >= 0)").unwrap()
    }
}

#[impl_stack]
impl<T: Float, D: MainMemory, S: Shape> ActivationOps<T, S, D> for CPU {
    #[inline]
    fn sigmoid(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        each_op(self, x, |x| T::one() / (T::one() + -x.exp()))
    }
    #[inline]
    fn sigmoid_grad(&self, activated: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        each_op(self, activated, |x| x * (T::one() - x))
    }

    #[inline]
    fn tanh(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        each_op(self, x, |x| x.tanh())
    }

    #[inline]
    fn tanh_grad(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        each_op(self, x, |x| T::one() - x.tanh().powi(2))
    }

    #[inline]
    fn relu(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        each_op(self, x, |x| T::from_usize((x >= T::zero()) as usize) * x)
    }

    #[inline]
    fn relu_mut(&self, x: &mut Matrix<T, D, S>) {
        each_op_slice_mut(x, |x| T::from_usize((x >= T::zero()) as usize) * x)
    }

    #[inline]
    fn relu_grad(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        each_op(self, x, |x| T::from_usize((x >= T::default()) as usize))
    }

    #[inline]
    fn relu_grad_mut(&self, x: &mut Matrix<T, D, S>) {
        each_op_slice_mut(x, |x| T::from_usize((x >= T::default()) as usize))
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> ActivationOps<T> for CUDA {
    #[inline]
    fn sigmoid(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        let out = cu_str_op(self, x, "1.0 / (1.0 + exp(-x))").unwrap();
        (out, x.dims()).into()
    }

    fn sigmoid_grad(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        let out = cu_str_op(self, x, "x * (1.0 - x)").unwrap();
        (out, x.dims()).into()
    }

    #[inline]
    fn tanh(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        let out = cu_str_op(self, x, "tanh(x)").unwrap();
        (out, x.dims()).into()
    }

    #[inline]
    fn tanh_grad(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        let out = cu_str_op(
            self,
            x,
            &format!(
                "{dtype} (1.0 - pow((double) tanh(x), 2.0))",
                dtype = T::as_c_type_str()
            ),
        )
        .unwrap();
        (out, x.dims()).into()
    }

    #[inline]
    fn relu(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        let out = cu_str_op(self, x, "x * (x >= 0)").unwrap();
        (out, x.dims()).into()
    }

    #[inline]
    fn relu_grad(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        let out = cu_str_op(self, x, "(x >= 0)").unwrap();
        (out, x.dims()).into()
    }

    #[inline]
    fn relu_mut(&self, x: &mut Matrix<T, Self, ()>) {
        cu_str_op_mut(self, x, "x * (x >= 0)").unwrap();
    }

    #[inline]
    fn relu_grad_mut(&self, x: &mut Matrix<T, Self, ()>) {
        cu_str_op_mut(self, x, "(x >= 0)").unwrap()
    }
}
