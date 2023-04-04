use crate::Matrix;
use custos::{number::Number, CDatatype, Device, MainMemory, CPU};

#[cfg(feature = "cpu")]
use custos::cache::Cache;

#[cfg(feature = "cuda")]
use crate::{cu_to_cpu_s, cu_to_cpu_scalar};
#[cfg(feature = "cuda")]
use custos::CUDA;

#[cfg(feature = "opencl")]
use super::{cl_to_cpu_s, cl_to_cpu_scalar};
#[cfg(feature = "opencl")]
use custos::OpenCL;

impl<'a, T, D: MaxOps<T>> Matrix<'a, T, D> {
    #[inline]
    pub fn max(&self) -> T {
        self.device().max(self)
    }

    #[inline]
    pub fn max_rows(&self) -> Matrix<'a, T, D> {
        self.device().max_rows(self)
    }

    #[inline]
    pub fn max_cols(&self) -> Matrix<'a, T, D> {
        self.device().max_cols(self)
    }
}

pub trait MaxOps<T, D: Device = Self>: Device {
    fn max(&self, x: &Matrix<T, D>) -> T;
    fn max_rows(&self, x: &Matrix<T, D>) -> Matrix<T, Self>;
    fn max_cols(&self, x: &Matrix<T, D>) -> Matrix<T, Self>;
}

// TODO: refactor this into own methods
#[cfg(feature = "cpu")]
impl<T: Copy + PartialOrd, D: MainMemory> MaxOps<T, D> for CPU {
    fn max(&self, x: &Matrix<T, D>) -> T {
        let mut max = x[0];

        for value in x.iter() {
            if *value > max {
                max = *value;
            }
        }
        max
    }

    fn max_rows(&self, x: &Matrix<T, D>) -> Matrix<T> {
        let mut out = self.retrieve(x.cols(), x.as_buf());

        let data = x.as_slice();
        let max_rows = out.as_mut_slice();

        max_rows.copy_from_slice(&data[..max_rows.len()]);

        for idx in 0..x.rows() {
            let index = idx * x.cols();
            let row = &data[index..index + x.cols()];

            for (i, data) in row.iter().enumerate() {
                if data > &max_rows[i] {
                    max_rows[i] = *data;
                }
            }
        }
        (out, 1, x.cols()).into()
    }

    fn max_cols(&self, x: &Matrix<T, D>) -> Matrix<T> {
        let data = x.as_slice();
        let mut out = self.retrieve(x.rows(), x.as_buf());
        
        let max_cols = out.as_mut_slice();

        for (idx, max_cols_val) in max_cols.iter_mut().enumerate().take(x.rows()) {
            let index = idx * x.cols();
            let row = &data[index..index + x.cols()];

            let mut max = row[0];

            for data in row {
                if data > &max {
                    max = *data;
                }
            }
            *max_cols_val = max;
        }
        (out, x.rows(), 1).into()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype + Number> MaxOps<T> for OpenCL {
    fn max(&self, x: &Matrix<T, Self>) -> T {
        cl_to_cpu_scalar(self, x, |device, x| device.max(x))
    }

    fn max_rows(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_to_cpu_s(self, x, |device, x| device.max_rows(x))
    }

    fn max_cols(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_to_cpu_s(self, x, |device, x| device.max_cols(x))
    }
}

#[cfg(feature = "cuda")]
impl<T: Number> MaxOps<T> for CUDA {
    fn max(&self, x: &Matrix<T, CUDA>) -> T {
        cu_to_cpu_scalar(x, |cpu, x| cpu.max(&x))
    }

    fn max_rows(&self, x: &Matrix<T, CUDA>) -> Matrix<T, CUDA> {
        cu_to_cpu_s(self, x, |cpu, x| cpu.max_rows(&x))
    }

    fn max_cols(&self, x: &Matrix<T, CUDA>) -> Matrix<T, CUDA> {
        cu_to_cpu_s(self, x, |cpu, x| cpu.max_cols(&x))
    }
}
