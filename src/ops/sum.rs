use crate::Matrix;
use custos::{get_device, number::Number, CDatatype, CPU, Cache};

#[cfg(feature = "opencl")]
use super::{cl_to_cpu_s, cl_to_cpu_scalar};
#[cfg(feature = "opencl")]
use custos::CLDevice;

#[cfg(feature = "cuda")]
use crate::{cu_to_cpu_s, cu_to_cpu_scalar};
#[cfg(feature = "cuda")]
use custos::CudaDevice;

impl<'a, T: CDatatype> Matrix<'a, T> {
    pub fn sum(&self) -> T {
        get_device!(self.device(), SumOps<T>).sum(self)
    }

    pub fn mean(&self) -> T {
        get_device!(self.device(), SumOps<T>).mean(self)
    }

    pub fn sum_rows(&self) -> Matrix<'a, T> {
        get_device!(self.device(), SumOps<T>).sum_rows(self)
    }

    pub fn sum_cols(&self) -> Matrix<'a, T> {
        get_device!(self.device(), SumOps<T>).sum_cols(self)
    }
}

pub trait SumOps<T> {
    fn sum(&self, x: &Matrix<T>) -> T;
    fn mean(&self, x: &Matrix<T>) -> T;
    fn sum_rows(&self, x: &Matrix<T>) -> Matrix<T>;
    fn sum_cols(&self, x: &Matrix<T>) -> Matrix<T>;
}

impl<T: Number> SumOps<T> for CPU {
    fn sum(&self, x: &Matrix<T>) -> T {
        x.iter().copied().sum()
        /*let mut sum = T::default();
        for value in x.as_slice() {
            sum += *value;
        }
        sum*/
    }

    fn mean(&self, x: &Matrix<T>) -> T {
        let sum = self.sum(x);
        sum / T::from_usize(x.size())
    }

    fn sum_rows(&self, x: &Matrix<T>) -> Matrix<T> {
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

    fn sum_cols(&self, x: &Matrix<T>) -> Matrix<T> {
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
impl<T: CDatatype> SumOps<T> for CLDevice {
    fn sum(&self, x: &Matrix<T>) -> T {
        cl_to_cpu_scalar(self, x, |device, x| device.sum(x))
    }

    fn mean(&self, x: &Matrix<T>) -> T {
        cl_to_cpu_scalar(self, x, |device, x| device.mean(x))
    }

    fn sum_rows<'a>(&'a self, x: &Matrix<T>) -> Matrix<'a, T> {
        cl_to_cpu_s(self, x, |device, x| device.sum_rows(x))
    }

    fn sum_cols(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_s(self, x, |device, x| device.sum_cols(x))
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> SumOps<T> for CudaDevice {
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
