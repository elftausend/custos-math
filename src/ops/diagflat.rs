#[cfg(feature = "cuda")]
use crate::cu_to_cpu_s;
use crate::Matrix;
#[cfg(feature = "cuda")]
use custos::CudaDevice;
use custos::{cache::Cache, cpu::CPU, get_device, CDatatype};

#[cfg(feature = "opencl")]
use super::cl_to_cpu_s;
#[cfg(feature = "opencl")]
use custos::CLDevice;

impl<'a, T: CDatatype> Matrix<'a, T> {
    pub fn diagflat(&self) -> Matrix<'a, T> {
        get_device!(self.device(), DiagflatOp<T>).diagflat(self)
    }
}

pub fn diagflat<T: Copy>(a: &[T], b: &mut [T]) {
    for (row, x) in a.iter().enumerate() {
        b[row * a.len() + row] = *x;
    }
}

pub trait DiagflatOp<T> {
    fn diagflat(&self, x: &Matrix<T>) -> Matrix<T>;
}

impl<T: Default + Copy> DiagflatOp<T> for CPU {
    fn diagflat(&self, x: &Matrix<T>) -> Matrix<T> {
        assert!(x.dims().0 == 1 || x.dims().1 == 1);
        let size = x.size();

        let mut y = Cache::get(self, size * size, x.node.idx);
        diagflat(x.as_slice(), y.as_mut_slice());
        (y, (size, size)).into()
    }
}

#[cfg(feature = "cuda")]
impl<T: Copy + Default> DiagflatOp<T> for CudaDevice {
    fn diagflat(&self, x: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_s(self, x, |cpu, x| cpu.diagflat(&x))
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> DiagflatOp<T> for CLDevice {
    fn diagflat(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_s(self, x, |device, x| device.diagflat(x))
    }
}