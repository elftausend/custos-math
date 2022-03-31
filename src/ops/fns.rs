use custos::{Matrix, cpu::{InternCPU, each_op}, number::Float, opencl::{GenericOCL, InternCLDevice}, get_device};

use crate::opencl::str_op;

pub trait Fns<T> {
    fn exp(&self) -> Matrix<T>;
    fn ln(&self) -> Matrix<T>;
    fn neg(&self) -> Matrix<T>;
}

impl <T: GenericOCL+Float>Fns<T> for Matrix<T> {
    fn exp(&self) -> Matrix<T> {
        let device = get_device!(FnsOps, T).unwrap();
        device.exp(*self)
    }

    fn ln(&self) -> Matrix<T> {
        let device = get_device!(FnsOps, T).unwrap();
        device.ln(*self)
    }

    fn neg(&self) -> Matrix<T> {
        let device = get_device!(FnsOps, T).unwrap();
        device.neg(*self)
    }
}

pub trait FnsOps<T> {
    fn exp(&self, x: Matrix<T>) -> Matrix<T>;
    fn ln(&self, x: Matrix<T>) -> Matrix<T>;
    fn neg(&self, x: Matrix<T>) -> Matrix<T>;
}

impl <T: Float>FnsOps<T> for InternCPU {
    fn exp(&self, x: Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.exp())
    }

    fn ln(&self, x: Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.ln())
    }

    fn neg(&self, x: Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.negate())
    }
}

impl <T: GenericOCL>FnsOps<T> for InternCLDevice {
    fn exp(&self, x: Matrix<T>) -> Matrix<T> {
        str_op(self.clone(), x, "exp(I)").unwrap()
    }

    fn ln(&self, x: Matrix<T>) -> Matrix<T> {
        str_op(self.clone(), x, "log(I)").unwrap()
    }

    fn neg(&self, x: Matrix<T>) -> Matrix<T> {
        str_op(self.clone(), x, "-I").unwrap()
    }
}