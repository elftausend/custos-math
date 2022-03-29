use custos::{Matrix, cpu::{InternCPU, CPUCache}, opencl::{GenericOCL, InternCLDevice}};

use super::switch_to_cpu_help_s;

pub trait DiagflatOp<T> {
    fn diagflat(&self, x: Matrix<T>) -> Matrix<T>;
}

impl <T: Default+Copy>DiagflatOp<T> for InternCPU {
    fn diagflat(&self, x: Matrix<T>) -> Matrix<T> {
        assert!(x.dims().0 == 1 || x.dims().1 == 1);
        let size = x.size();
        
        let mut y = CPUCache::get::<T>(self.clone(), (size, size));

        for idx in 0..size {
            let index = idx*size;
            let row = &mut y.as_cpu_slice_mut()[index..index+size];
            row[idx] = x.as_cpu_slice()[idx];
        }
        y
    }
}

impl <T: GenericOCL>DiagflatOp<T> for InternCLDevice {
    fn diagflat(&self, x: Matrix<T>) -> Matrix<T> {
        switch_to_cpu_help_s(self, x, |device, x| device.diagflat(x))
    }
}

