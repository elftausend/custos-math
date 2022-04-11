use custos::{Matrix, InternCPU, number::Number, cpu::CPUCache, InternCLDevice, get_device, GenericOCL};

use super::{switch_to_cpu_help_scalar, switch_to_cpu_help_s};

pub trait Sum<T> {
    fn sum(&self) -> T;
    fn mean(&self) -> T;
    fn sum_rows(&self) -> Matrix<T>;
    fn sum_cols(&self) -> Matrix<T>;
}

impl <T: GenericOCL>Sum<T> for Matrix<T> {
    fn sum(&self) -> T {
        let device = get_device!(SumOps, T).unwrap();
        device.sum(self)
    }

    fn mean(&self) -> T {
        let device = get_device!(SumOps, T).unwrap();
        device.mean(self)
    }

    fn sum_rows(&self) -> Matrix<T> {
        let device = get_device!(SumOps, T).unwrap();
        device.sum_rows(self)
    }

    fn sum_cols(&self) -> Matrix<T> {
        let device = get_device!(SumOps, T).unwrap();
        device.sum_cols(self)
    }
}

pub trait SumOps<T> {
    fn sum(&self, x: &Matrix<T>) -> T;
    fn mean(&self, x: &Matrix<T>) -> T;
    fn sum_rows(&self, x: &Matrix<T>) -> Matrix<T>;
    fn sum_cols(&self, x: &Matrix<T>) -> Matrix<T>;
}

impl <T: Number>SumOps<T> for InternCPU {
    fn sum(&self, x: &Matrix<T>) -> T {
        let mut sum = T::default();
        for value in x.as_cpu_slice() {
            sum += *value;
        }
        sum
    }

    fn mean(&self, x: &Matrix<T>) -> T {
        let sum = self.sum(x);
        sum/T::from_usize(x.size())
    }

    fn sum_rows(&self, x: &Matrix<T>) -> Matrix<T> {
        let mut y = CPUCache::get(self.clone(), (1, x.cols()));

        let data = x.as_cpu_slice();
        let sum_slice = y.as_cpu_slice_mut();

        for value in sum_slice.iter_mut() {
            *value = T::default();
        }

        for idx in 0..x.rows() {
            let index = idx*x.cols();
            let row = &data[index..index+x.cols()];

            for (i, value) in row.iter().enumerate() {
                sum_slice[i] += *value;
            }
        }
        y
    }

    fn sum_cols(&self, x: &Matrix<T>) -> Matrix<T> {
        let mut y = CPUCache::get(self.clone(), (x.rows(), 1));
 
        let data = x.as_cpu_slice();
        let sum_slice = y.as_cpu_slice_mut();
        
        for (idx, col_vec_value) in sum_slice.iter_mut().enumerate().take(x.rows()) {
            let index = idx*x.cols();
            let row = &data[index..index+x.cols()];
            let mut sum = T::default();
            
            for data in row {
                sum += *data;
            }
            *col_vec_value = sum;
        }
        y
    }
}

impl <T: GenericOCL>SumOps<T> for InternCLDevice {
    fn sum(&self, x: &Matrix<T>) -> T {
        switch_to_cpu_help_scalar(self, x, |device, x| device.sum(&x))
    }

    fn mean(&self, x: &Matrix<T>) -> T {
        switch_to_cpu_help_scalar(self, x, |device, x| device.mean(&x))
    }

    fn sum_rows(&self, x: &Matrix<T>) -> Matrix<T> {
        switch_to_cpu_help_s(self, x, |device, x| device.sum_rows(&x))
    }

    fn sum_cols(&self, x: &Matrix<T>) -> Matrix<T> {
        switch_to_cpu_help_s(self, x, |device, x| device.sum_cols(&x))
    }
}