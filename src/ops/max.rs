use crate::Matrix;
use custos::{cache::Cache, number::Number, CDatatype, Device, MainMemory, CPU};

#[cfg(feature = "cuda")]
use crate::{cu_to_cpu_s, cu_to_cpu_scalar};
#[cfg(feature = "cuda")]
use custos::CUDA;

#[cfg(feature = "opencl")]
use super::{cl_to_cpu_s, cl_to_cpu_scalar};
#[cfg(feature = "opencl")]
use custos::OpenCL;

impl<'a, T: CDatatype, D: MaxOps<T>> Matrix<'a, T, D> {
    pub fn max(&self) -> T {
        self.device().max(self)
    }

    pub fn max_rows(&self) -> Matrix<'a, T, D> {
        self.device().max_rows(self)
    }

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
impl<T: Number, D: MainMemory> MaxOps<T, D> for CPU {
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
        let mut y = Cache::get(self, x.cols(), x.node.idx);

        let data = x.as_slice();
        let max_rows = y.as_mut_slice();

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
        (y, 1, x.cols()).into()
    }

    fn max_cols(&self, x: &Matrix<T, D>) -> Matrix<T> {
        let data = x.as_slice();
        let mut y = Cache::get(self, x.rows(), x.node.idx);

        let max_cols = y.as_mut_slice();

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
        (y, x.rows(), 1).into()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> MaxOps<T> for OpenCL {
    fn max(&self, x: &Matrix<T, Self>) -> T {
        cl_to_cpu_scalar(x, |device, x| device.max(x))
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
