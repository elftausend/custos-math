use custos::{cpu::CPU, number::Float, CDatatype, Device, MainMemory, Shape, impl_stack};

#[cfg(feature="stack")]
use custos::Stack;

use crate::{each_op, Matrix};

#[cfg(feature = "cuda")]
use crate::cu_str_op;
#[cfg(feature = "cuda")]
use custos::CUDA;

#[cfg(feature = "opencl")]
use crate::opencl::cl_str_op_mat;
#[cfg(feature = "opencl")]
use custos::OpenCL;

impl<'a, T: CDatatype + Float, S: Shape, D: FnsOps<T, D, S>> Matrix<'a, T, D, S> {
    pub fn exp(&self) -> Self {
        self.device().exp(self)
    }

    pub fn ln(&self) -> Self {
        self.device().ln(self)
    }

    pub fn neg(&self) -> Self {
        self.device().neg(self)
    }

    pub fn powf(&self, rhs: T) -> Self {
        self.device().powf(self, rhs)
    }

    pub fn powi(&self, rhs: i32) -> Self {
        self.device().powi(self, rhs)
    }
}

pub trait FnsOps<T, D: Device = Self, S: Shape = ()>: Device {
    fn exp(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S>;
    fn ln(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S>;
    fn neg(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S>;
    fn powf(&self, x: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S>;
    fn powi(&self, x: &Matrix<T, D, S>, rhs: i32) -> Matrix<T, Self, S>;
}

#[impl_stack]
impl<T, D, S> FnsOps<T, D, S> for CPU
where
    T: Float,
    D: MainMemory,
    S: Shape,
{
    fn exp(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        each_op(self, x, |x| x.exp())
    }

    fn ln(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        each_op(self, x, |x| x.ln())
    }

    fn neg(&self, x: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        each_op(self, x, |x| -x)
    }

    fn powf(&self, x: &Matrix<T, D, S>, rhs: T) -> Matrix<T, Self, S> {
        each_op(self, x, |x| x.powf(rhs))
    }

    fn powi(&self, x: &Matrix<T, D, S>, rhs: i32) -> Matrix<T, Self, S> {
        each_op(self, x, |x| x.powi(rhs))
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> FnsOps<T> for OpenCL {
    fn exp(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, "exp(x)").unwrap()
    }

    fn ln(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, "log(x)").unwrap()
    }

    fn neg(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, "-x").unwrap()
    }

    fn powf(&self, x: &Matrix<T, Self>, rhs: T) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, &format!("pow(x, {rhs})")).unwrap()
    }

    fn powi(&self, x: &Matrix<T, Self>, rhs: i32) -> Matrix<T, Self> {
        cl_str_op_mat(self, x, &format!("pow(x, {rhs})")).unwrap()
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> FnsOps<T> for CUDA {
    fn exp(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        let out = cu_str_op(self, x, "exp(x)").unwrap();
        (out, x.dims()).into()
    }

    fn ln(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        let out = cu_str_op(self, x, "logf(x)").unwrap();
        (out, x.dims()).into()
    }

    fn neg(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        let out = cu_str_op(self, x, "-x").unwrap();
        (out, x.dims()).into()
    }

    fn powf(&self, x: &Matrix<T, Self>, rhs: T) -> Matrix<T, Self> {
        let out = cu_str_op(self, x, &format!("powf(x, {rhs})")).unwrap();
        (out, x.dims()).into()
    }

    fn powi(&self, x: &Matrix<T, Self>, rhs: i32) -> Matrix<T, Self> {
        let out = cu_str_op(self, x, &format!("powf(x, {rhs})")).unwrap();
        (out, x.dims()).into()
    }
}


#[cfg(test)]
mod tests {

    #[cfg(feature="stack")]
    #[test]
    fn test_stack_impl() {
        use custos::{Buffer, Stack};

        use crate::Matrix;

        let data = Buffer::from((Stack, &[3., 1., 5.,]));
        let mat = Matrix {
            data,
            dims: (1, 3),
        };

        mat.ln();
    }
}