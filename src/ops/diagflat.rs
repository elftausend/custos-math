use custos::{Matrix, cpu::{InternCPU, CPUCache}, opencl::{GenericOCL, InternCLDevice}, get_device};

use super::switch_to_cpu_help_s;

pub trait Diagflat<T> {
    fn diagflat(&self) -> Matrix<T>;
}

impl <T: GenericOCL>Diagflat<T> for Matrix<T> {
    fn diagflat(&self) -> Matrix<T> {
        let device = get_device!(DiagflatOp, T).unwrap();
        device.diagflat(*self)
    }
}

pub fn diagflat<T: Copy>(size: usize, a: &[T], b: &mut [T]) {
    for (row, a) in a.iter().enumerate() {
        let index = row*size+row;
        b[index] = *a;
    }
}

pub trait DiagflatOp<T> {
    fn diagflat(&self, x: Matrix<T>) -> Matrix<T>;
}

impl <T: Default+Copy>DiagflatOp<T> for InternCPU {
    fn diagflat(&self, x: Matrix<T>) -> Matrix<T> {
        assert!(x.dims().0 == 1 || x.dims().1 == 1);
        let size = x.size();
        
        let mut y = CPUCache::get::<T>(self.clone(), (size, size));

        diagflat(size, x.as_cpu_slice(), y.as_cpu_slice_mut());
        y
    }
}

impl <T: GenericOCL>DiagflatOp<T> for InternCLDevice {
    fn diagflat(&self, x: Matrix<T>) -> Matrix<T> {
        switch_to_cpu_help_s(self, x, |device, x| device.diagflat(x))
    }
}

