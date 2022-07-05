use custos::{
    cpu::{each_op, CPU},
    get_device,
    number::Float,
    opencl::CLDevice,
    CDatatype, Matrix, CudaDevice,
};

use crate::{opencl::cl_str_op, cu_str_op};

pub trait Fns<T> {
    #[must_use]
    fn exp(&self) -> Matrix<T>;
    #[must_use]
    fn ln(&self) -> Matrix<T>;
    #[must_use]
    fn neg(&self) -> Matrix<T>;
    #[must_use]
    fn powf(&self, rhs: T) -> Matrix<T>;
}

impl<T: CDatatype + Float> Fns<T> for Matrix<T> {
    fn exp(&self) -> Matrix<T> {
        let device = get_device!(FnsOps, T).unwrap();
        device.exp(self)
    }

    fn ln(&self) -> Matrix<T> {
        let device = get_device!(FnsOps, T).unwrap();
        device.ln(self)
    }

    fn neg(&self) -> Matrix<T> {
        let device = get_device!(FnsOps, T).unwrap();
        device.neg(self)
    }

    fn powf(&self, rhs: T) -> Matrix<T> {
        let device = get_device!(FnsOps, T).unwrap();
        device.powf(self, rhs)
    }
}

pub trait FnsOps<T> {
    fn exp(&self, x: &Matrix<T>) -> Matrix<T>;
    fn ln(&self, x: &Matrix<T>) -> Matrix<T>;
    fn neg(&self, x: &Matrix<T>) -> Matrix<T>;
    fn powf(&self, x: &Matrix<T>, rhs: T) -> Matrix<T>; 
    
}

impl<T: Float> FnsOps<T> for CPU {
    fn exp(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.exp())
    }

    fn ln(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.ln())
    }

    fn neg(&self, x: &Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.negate())
    }

    fn powf(&self, x: &Matrix<T>, rhs: T) -> Matrix<T> {
        each_op(self, x, |x| x.powf(rhs))
    }
}

impl<T: CDatatype> FnsOps<T> for CLDevice {
    fn exp(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_str_op(self, x, "exp(x)").unwrap()
    }

    fn ln(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_str_op(self, x, "log(x)").unwrap()
    }

    fn neg(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_str_op(self, x, "-x").unwrap()
    }

    fn powf(&self, x: &Matrix<T>, rhs: T) -> Matrix<T> {
        cl_str_op(self, x, &format!("pow(x, {rhs})")).unwrap()
    }
}

impl<T: CDatatype> FnsOps<T> for CudaDevice {
    fn exp(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "exp(x)").unwrap();
        (out, x.dims()).into()
    }

    fn ln(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "ln(x)").unwrap();
        (out, x.dims()).into()
    }

    fn neg(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = cu_str_op(self, x, "-x").unwrap();
        (out, x.dims()).into()
    }

    fn powf(&self, x: &Matrix<T>, rhs: T) -> Matrix<T> {
        let out = cu_str_op(self, x, &format!("pow(x, {rhs})")).unwrap();
        (out, x.dims()).into()
    }
}