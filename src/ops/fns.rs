use custos::{cpu::CPU, number::Float, CDatatype, Device, MainMemory};

use crate::{each_op, Matrix};

#[cfg(feature = "cuda")]
use crate::cu_str_op;
#[cfg(feature = "cuda")]
use custos::CudaDevice;

#[cfg(feature = "opencl")]
use crate::opencl::cl_str_op_mat;
#[cfg(feature = "opencl")]
use custos::OpenCL;

impl<'a, T: CDatatype + Float, D: FnsOps<T>> Matrix<'a, T, D> {
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

pub trait FnsOps<T, D: Device = Self>: Device {
    fn exp(&self, x: &Matrix<T, D>) -> Matrix<T, Self>;
    fn ln(&self, x: &Matrix<T, D>) -> Matrix<T, Self>;
    fn neg(&self, x: &Matrix<T, D>) -> Matrix<T, Self>;
    fn powf(&self, x: &Matrix<T, D>, rhs: T) -> Matrix<T, Self>;
    fn powi(&self, x: &Matrix<T, D>, rhs: i32) -> Matrix<T, Self>;
}

impl<T: Float, D: MainMemory> FnsOps<T, D> for CPU {
    fn exp(&self, x: &Matrix<T, D>) -> Matrix<T> {
        each_op(self, x, |x| x.exp())
    }

    fn ln(&self, x: &Matrix<T, D>) -> Matrix<T> {
        each_op(self, x, |x| x.ln())
    }

    fn neg(&self, x: &Matrix<T, D>) -> Matrix<T> {
        each_op(self, x, |x| -x)
    }

    fn powf(&self, x: &Matrix<T, D>, rhs: T) -> Matrix<T> {
        each_op(self, x, |x| x.powf(rhs))
    }

    fn powi(&self, x: &Matrix<T, D>, rhs: i32) -> Matrix<T> {
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
impl<T: CDatatype> FnsOps<T> for CudaDevice {
    fn exp(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
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
