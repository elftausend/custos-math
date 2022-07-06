use custos::{
    cpu::CPUCache, get_device, number::Number, CDatatype, CLDevice, CPU, Matrix,
};
#[cfg(feature="cuda")]
use custos::CudaDevice;
#[cfg(feature="cuda")]
use crate::{cu_to_cpu_s, cu_to_cpu_scalar};

use super::{cl_to_cpu_s, cl_to_cpu_scalar};

pub trait Max<T> {
    fn max(&self) -> T;
    fn max_rows(&self) -> Matrix<T>;
    fn max_cols(&self) -> Matrix<T>;
}

impl<T: CDatatype> Max<T> for Matrix<T> {
    fn max(&self) -> T {
        let device = get_device!(MaxOps, T).unwrap();
        device.max(self)
    }

    fn max_rows(&self) -> Matrix<T> {
        let device = get_device!(MaxOps, T).unwrap();
        device.max_rows(self)
    }

    fn max_cols(&self) -> Matrix<T> {
        let device = get_device!(MaxOps, T).unwrap();
        device.max_cols(self)
    }
}

pub trait MaxOps<T> {
    fn max(&self, x: &Matrix<T>) -> T;
    fn max_rows(&self, x: &Matrix<T>) -> Matrix<T>;
    fn max_cols(&self, x: &Matrix<T>) -> Matrix<T>;
}

impl<T: Number> MaxOps<T> for CPU {
    fn max(&self, x: &Matrix<T>) -> T {
        let slice = x.as_slice();
        let mut max = slice[0];

        for value in slice {
            if *value > max {
                max = *value;
            }
        }
        max
    }

    fn max_rows(&self, x: &Matrix<T>) -> Matrix<T> {
        let mut y = CPUCache::get::<T>(self, x.cols());

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

    fn max_cols(&self, x: &Matrix<T>) -> Matrix<T> {
        let data = x.as_slice();
        let mut y = CPUCache::get::<T>(self, x.rows());

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

impl<T: CDatatype> MaxOps<T> for CLDevice {
    fn max(&self, x: &Matrix<T>) -> T {
        cl_to_cpu_scalar(self, x, |device, x| device.max(&x))
    }

    fn max_rows(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_s(self, x, |device, x| device.max_rows(&x))
    }

    fn max_cols(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_s(self, x, |device, x| device.max_cols(&x))
    }
}

#[cfg(feature="cuda")]
impl<T: Number> MaxOps<T> for CudaDevice {
    fn max(&self, x: &Matrix<T>) -> T {
        cu_to_cpu_scalar(self, x, |cpu, x| cpu.max(&x))
    }

    fn max_rows(&self, x: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_s(self, x, |cpu, x| cpu.max_rows(&x))
    }

    fn max_cols(&self, x: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_s(self, x, |cpu, x| cpu.max_cols(&x))
    }
}
