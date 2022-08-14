use crate::{cpu::row_op, row_op_slice_lhs, Matrix};
use custos::{cpu::CPU, get_device, number::Number, CDatatype};

#[cfg(feature = "opencl")]
use crate::{cl_to_cpu_lr, opencl};
#[cfg(feature = "opencl")]
use custos::CLDevice;

#[cfg(feature = "cuda")]
use crate::{cu_to_cpu_lr, cu_to_cpu_lr_mut};
#[cfg(feature = "cuda")]
use custos::CudaDevice;

impl<'a, T: CDatatype> Matrix<'a, T> {
    pub fn add_row(&self, rhs: &Matrix<T>) -> Matrix<'a, T> {
        let device = get_device!(self.device(), RowOp<T>);
        device.add_row(self, rhs)
    }

    pub fn add_row_mut(&mut self, rhs: &Matrix<'a, T>) {
        let device = get_device!(self.device(), RowOp<T>);
        device.add_row_mut(self, rhs)
    }
}

pub trait RowOp<T> {
    fn add_row(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
    fn add_row_mut(&self, lhs: &mut Matrix<T>, rhs: &Matrix<T>);
}

impl<T: Number> RowOp<T> for CPU {
    fn add_row(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        row_op(self, lhs, rhs, |c, a, b| *c = a + b)
    }

    fn add_row_mut(&self, lhs: &mut Matrix<T>, rhs: &Matrix<T>) {
        let (lhs_rows, lhs_cols) = lhs.dims();
        row_op_slice_lhs(lhs, lhs_rows, lhs_cols, rhs, |c, a| *c += a)
        
    }
}

// TODO: Implement add_ro_mut (for cuda as well)
#[cfg(feature = "opencl")]
impl<T: CDatatype> RowOp<T> for CLDevice {
    fn add_row(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.add_row(lhs, rhs))
    }

    fn add_row_mut(&self, lhs: &mut Matrix<T>, rhs: &Matrix<T>) {
        opencl::cpu_exec_lhs_rhs_mut(self, lhs, rhs, |cpu, lhs, rhs| cpu.add_row_mut(lhs, rhs))
            .unwrap();
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> RowOp<T> for CudaDevice {
    fn add_row(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.add_row(lhs, rhs))
    }

    fn add_row_mut(&self, lhs: &mut Matrix<T>, rhs: &Matrix<T>) {
        cu_to_cpu_lr_mut(self, lhs, rhs, |device, lhs, rhs| {
            device.add_row_mut(lhs, rhs)
        })
    }
}
