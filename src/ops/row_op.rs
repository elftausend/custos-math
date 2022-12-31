use crate::{cpu::row_op, row_op_slice_lhs, Matrix};
use custos::{number::Number, CDatatype, Device, MainMemory};

#[cfg(feature="cpu")]
use custos::CPU;

#[cfg(feature = "opencl")]
use crate::{cl_to_cpu_lr, opencl};
#[cfg(feature = "opencl")]
use custos::OpenCL;

#[cfg(feature = "cuda")]
use crate::{cu_to_cpu_lr, cu_to_cpu_lr_mut};
#[cfg(feature = "cuda")]
use custos::CUDA;

impl<'a, T: CDatatype, D: RowOp<T>> Matrix<'a, T, D> {
    #[inline]
    pub fn add_row(&self, rhs: &Matrix<T, D>) -> Matrix<'a, T, D> {
        self.device().add_row(self, rhs)
    }

    #[inline]
    pub fn add_row_mut(&mut self, rhs: &Matrix<'a, T, D>) {
        rhs.device().add_row_mut(self, rhs)
    }
}

pub trait RowOp<T, D: Device = Self>: Device {
    fn add_row(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T, Self>;
    fn add_row_mut(&self, lhs: &mut Matrix<T, D>, rhs: &Matrix<T, D>);
}

#[cfg(feature="cpu")]
impl<T: Number, D: MainMemory> RowOp<T, D> for CPU {
    #[inline]
    fn add_row(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T> {
        row_op(self, lhs, rhs, |c, a, b| *c = a + b)
    }

    #[inline]
    fn add_row_mut(&self, lhs: &mut Matrix<T, D>, rhs: &Matrix<T, D>) {
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
