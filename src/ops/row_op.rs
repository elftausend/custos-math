use crate::{cpu::row_op, row_op_slice_lhs, Matrix};
use custos::{impl_stack, number::Number, CDatatype, Device, MainMemory, Shape};

#[cfg(feature = "stack")]
use custos::Stack;

#[cfg(feature = "cpu")]
use custos::CPU;

#[cfg(feature = "opencl")]
use crate::{cl_to_cpu_lr, opencl};
#[cfg(feature = "opencl")]
use custos::OpenCL;

#[cfg(feature = "cuda")]
use crate::{cu_to_cpu_lr, cu_to_cpu_lr_mut};
#[cfg(feature = "cuda")]
use custos::CUDA;

impl<'a, T: CDatatype, LS: Shape, D: Device> Matrix<'a, T, D, LS> {
    #[inline]
    pub fn add_row<RS: Shape>(&self, rhs: &Matrix<T, D, RS>) -> Matrix<'a, T, D, LS>
    where
        D: RowOp<T, LS, RS>,
    {
        self.device().add_row(self, rhs)
    }

    #[inline]
    pub fn add_row_mut<RS: Shape>(&mut self, rhs: &Matrix<'a, T, D, RS>)
    where
        D: RowOp<T, LS, RS>,
    {
        rhs.device().add_row_mut(self, rhs)
    }
}

pub trait RowOp<T, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
    fn add_row(&self, lhs: &Matrix<T, D, LS>, rhs: &Matrix<T, D, RS>) -> Matrix<T, Self, LS>;
    fn add_row_mut(&self, lhs: &mut Matrix<T, D, LS>, rhs: &Matrix<T, D, RS>);
}

//#[cfg(feature = "cpu")]
#[impl_stack]
impl<T: Number, D: MainMemory, LS: Shape, RS: Shape> RowOp<T, LS, RS, D> for CPU {
    #[inline]
    fn add_row(&self, lhs: &Matrix<T, D, LS>, rhs: &Matrix<T, D, RS>) -> Matrix<T, Self, LS> {
        row_op(self, lhs, rhs, |c, a, b| *c = a + b)
    }

    #[inline]
    fn add_row_mut(&self, lhs: &mut Matrix<T, D, LS>, rhs: &Matrix<T, D, RS>) {
        let (lhs_rows, lhs_cols) = lhs.dims();
        row_op_slice_lhs(lhs, lhs_rows, lhs_cols, rhs, |c, a| *c += a)
    }
}

// TODO: Implement add_ro_mut (for cuda as well)
#[cfg(feature = "opencl")]
impl<T: CDatatype> RowOp<T> for OpenCL {
    #[inline]
    fn add_row(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.add_row(lhs, rhs))
    }

    #[inline]
    fn add_row_mut(&self, lhs: &mut Matrix<T, Self>, rhs: &Matrix<T, Self>) {
        opencl::cpu_exec_lhs_rhs_mut(self, lhs, rhs, |cpu, lhs, rhs| cpu.add_row_mut(lhs, rhs))
            .unwrap();
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> RowOp<T> for CUDA {
    #[inline]
    fn add_row(&self, lhs: &Matrix<T, CUDA>, rhs: &Matrix<T, CUDA>) -> Matrix<T, CUDA> {
        cu_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.add_row(lhs, rhs))
    }

    #[inline]
    fn add_row_mut(&self, lhs: &mut Matrix<T, CUDA>, rhs: &Matrix<T, CUDA>) {
        cu_to_cpu_lr_mut(self, lhs, rhs, |device, lhs, rhs| {
            device.add_row_mut(lhs, rhs)
        })
    }
}
