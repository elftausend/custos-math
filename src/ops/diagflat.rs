#[cfg(feature = "cuda")]
use crate::cu_to_cpu_s;
use crate::Matrix;
#[cfg(feature = "cuda")]
use custos::CUDA;
use custos::{CDatatype, Device, MainMemory};

#[cfg(feature = "cpu")]
use custos::{cache::Cache, cpu::CPU};

#[cfg(feature = "opencl")]
use super::cl_to_cpu_s;
#[cfg(feature = "opencl")]
use custos::OpenCL;

impl<'a, T: CDatatype, D: DiagflatOp<T>> Matrix<'a, T, D> {
    pub fn diagflat(&self) -> Matrix<'a, T, D> {
        self.device().diagflat(self)
    }
}

pub fn diagflat<T: Copy>(a: &[T], b: &mut [T]) {
    for (row, x) in a.iter().enumerate() {
        b[row * a.len() + row] = *x;
    }
}

pub trait DiagflatOp<T, D: Device = Self>: Device {
    fn diagflat(&self, x: &Matrix<T, D>) -> Matrix<T, Self>;
}

#[cfg(feature = "cpu")]
impl<T: Default + Copy, D: MainMemory> DiagflatOp<T, D> for CPU {
    fn diagflat(&self, x: &Matrix<T, D>) -> Matrix<T> {
        assert!(x.dims().0 == 1 || x.dims().1 == 1);
        let size = x.size();

        let mut out = Cache::get(self, size * size, x.node.idx);
        diagflat(x, &mut out);
        (out, (size, size)).into()
    }
}

#[cfg(feature = "cuda")]
impl<T: Copy + Default> DiagflatOp<T> for CUDA {
    #[inline]
    fn diagflat(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cu_to_cpu_s(self, x, |cpu, x| cpu.diagflat(&x))
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> DiagflatOp<T> for OpenCL {
    #[inline]
    fn diagflat(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_to_cpu_s(self, x, |device, x| device.diagflat(x))
    }
}
