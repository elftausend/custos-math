use crate::Matrix;
use custos::{number::Number, CDatatype, Cache, Device, MainMemory, CPU};

#[cfg(feature = "opencl")]
use super::{cl_to_cpu_s, cl_to_cpu_scalar};
#[cfg(feature = "opencl")]
use custos::OpenCL;

#[cfg(feature = "cuda")]
use crate::{cu_to_cpu_s, cu_to_cpu_scalar};
#[cfg(feature = "cuda")]
use custos::CUDA;

impl<'a, T: CDatatype, D: SumOps<T>> Matrix<'a, T, D> {
    pub fn sum(&self) -> T {
        self.device().sum(self)
    }

    pub fn mean(&self) -> T {
        self.device().mean(self)
    }

    pub fn sum_rows(&self) -> Matrix<'a, T, D> {
        self.device().sum_rows(self)
    }

    pub fn sum_cols(&self) -> Matrix<'a, T, D> {
        self.device().sum_cols(self)
    }
}

pub trait SumOps<T, D: Device = Self>: Device {
    fn sum(&self, x: &Matrix<T, D>) -> T;
    fn mean(&self, x: &Matrix<T, D>) -> T;
    fn sum_rows(&self, x: &Matrix<T, D>) -> Matrix<T, Self>;
    fn sum_cols(&self, x: &Matrix<T, D>) -> Matrix<T, Self>;
}

impl<T: Number, D: MainMemory> SumOps<T, D> for CPU {
    fn sum(&self, x: &Matrix<T, D>) -> T {
        x.iter().copied().sum()
        /*let mut sum = T::default();
        for value in x.as_slice() {
            sum += *value;
        }
        sum*/
    }

    fn mean(&self, x: &Matrix<T, D>) -> T {
        let sum = self.sum(x);
        sum / T::from_usize(x.size())
    }

    fn sum_rows(&self, x: &Matrix<T, D>) -> Matrix<T> {
        let mut out = Cache::get(self, x.cols(), x.node.idx);

        let data = x.as_slice();
        let sum_slice = out.as_mut_slice();

        for value in sum_slice.iter_mut() {
            *value = T::default();
        }

        for idx in 0..x.rows() {
            let index = idx * x.cols();
            let row = &data[index..index + x.cols()];

            for (i, value) in row.iter().enumerate() {
                sum_slice[i] += *value;
            }
        }
        (out, 1, x.cols()).into()
    }

    fn sum_cols(&self, x: &Matrix<T, D>) -> Matrix<T> {
        let mut out = Cache::get(self, x.rows(), x.node.idx);

        let data = x.as_slice();
        let sum_slice = out.as_mut_slice();

        for (idx, col_vec_value) in sum_slice.iter_mut().enumerate().take(x.rows()) {
            let index = idx * x.cols();
            let row = &data[index..index + x.cols()];
            let mut sum = T::default();

            for data in row {
                sum += *data;
            }
            *col_vec_value = sum;
        }
        (out, x.rows(), 1).into()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> SumOps<T> for OpenCL {
    fn sum(&self, x: &Matrix<T, Self>) -> T {
        cl_to_cpu_scalar(self, x, |device, x| device.sum(x))
    }

    fn mean(&self, x: &Matrix<T, Self>) -> T {
        cl_to_cpu_scalar(self, x, |device, x| device.mean(x))
    }

    fn sum_rows<'a>(&'a self, x: &Matrix<T, Self>) -> Matrix<'a, T, Self> {
        cl_to_cpu_s(self, x, |device, x| device.sum_rows(x))
    }

    fn sum_cols(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_to_cpu_s(self, x, |device, x| device.sum_cols(x))
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> SumOps<T> for CUDA {
    fn sum(&self, x: &Matrix<T>) -> T {
        cu_to_cpu_scalar(self, x, |device, x| device.sum(&x))
    }

    fn mean(&self, x: &Matrix<T>) -> T {
        cu_to_cpu_scalar(self, x, |device, x| device.mean(&x))
    }

    fn sum_rows(&self, x: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_s(self, x, |device, x| device.sum_rows(&x))
    }

    fn sum_cols(&self, x: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_s(self, x, |device, x| device.sum_cols(&x))
    }
}
