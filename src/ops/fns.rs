use custos::{cpu::CPU, get_device, number::Float, CDatatype};

use crate::{each_op, Matrix};

#[cfg(feature = "cuda")]
use crate::cu_str_op;
#[cfg(feature = "cuda")]
use custos::CudaDevice;

#[cfg(feature = "opencl")]
use crate::opencl::cl_str_op_mat;
#[cfg(feature = "opencl")]
use custos::CLDevice;

impl<'a, T: CDatatype + Float> Matrix<'a, T> {
    pub fn exp(&self) -> Matrix<'a, T> {
        get_device!(self.device(), FnsOps<T>).exp(self)
    }

    pub fn ln(&self) -> Matrix<'a, T> {
        get_device!(self.device(), FnsOps<T>).ln(self)
    }

    pub fn neg(&self) -> Matrix<'a, T> {
        get_device!(self.device(), FnsOps<T>).neg(self)
    }

    pub fn powf(&self, rhs: T) -> Matrix<'a, T> {
        get_device!(self.device(), FnsOps<T>).powf(self, rhs)
    }

    pub fn powi(&self, rhs: i32) -> Matrix<'a, T> {
        get_device!(self.device(), FnsOps<T>).powi(self, rhs)
    }
}

pub trait FnsOps<T> {
    fn exp(&self, x: &Matrix<T>) -> Matrix<T>;
    fn ln(&self, x: &Matrix<T>) -> Matrix<T>;
    fn neg(&self, x: &Matrix<T>) -> Matrix<T>;
    fn powf(&self, x: &Matrix<T>, rhs: T) -> Matrix<T>;
    fn powi(&self, x: &Matrix<T>, rhs: i32) -> Matrix<T>;
}

impl<T: Float> FnsOps<T> for CPU {
    fn exp(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.exp())
    }

    fn ln(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.ln())
    }

    fn neg(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| -x)
    }

    fn powf(&self, x: &Matrix<T>, rhs: T) -> Matrix<T> {
        each_op(self, x, |x| x.powf(rhs))
    }

    fn powi(&self, x: &Matrix<T>, rhs: i32) -> Matrix<T> {
        each_op(self, x, |x| x.powi(rhs))
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> FnsOps<T> for CLDevice {
    fn exp(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_str_op_mat(self, x, "exp(x)").unwrap()
    }

    fn ln(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_str_op_mat(self, x, "log(x)").unwrap()
    }

    fn neg(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_str_op_mat(self, x, "-x").unwrap()
    }

    fn powf(&self, x: &Matrix<T>, rhs: T) -> Matrix<T> {
        cl_str_op_mat(self, x, &format!("pow(x, {rhs})")).unwrap()
    }

    fn powi(&self, x: &Matrix<T>, rhs: i32) -> Matrix<T> {
        cl_str_op_mat(self, x, &format!("pow(x, {rhs})")).unwrap()
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> FnsOps<T> for CudaDevice {
    fn exp(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "exp(x)").unwrap();
        (out, x.dims()).into()
    }

    fn ln(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "logf(x)").unwrap();
        (out, x.dims()).into()
    }

    fn neg(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "-x").unwrap();
        (out, x.dims()).into()
    }

    fn powf(&self, x: &Matrix<T>, rhs: T) -> Matrix<T> {
        let out = cu_str_op(self, x, &format!("powf(x, {rhs})")).unwrap();
        (out, x.dims()).into()
    }

    fn powi(&self, x: &Matrix<T>, rhs: i32) -> Matrix<T> {
        let out = cu_str_op(self, x, &format!("powf(x, {rhs})")).unwrap();
        (out, x.dims()).into()
    }
}
