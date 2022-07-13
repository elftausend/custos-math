use custos::{
    get_device, number::Float, Device, CPU,
};
use crate::Matrix;
#[cfg(feature="opencl")]
use custos::CLDevice;
use rand::{thread_rng, Rng, distributions::uniform::SampleUniform};

#[cfg(feature="opencl")]
use crate::opencl::cl_write;

pub trait RandMatrix<T> {
    fn rand(&mut self, lo: T, hi: T);
}
impl<T: Float + SampleUniform> RandMatrix<T> for Matrix<T> {
    fn rand(&mut self, lo: T, hi: T) {
        let device = get_device!(RandOp<T>).unwrap();
        device.rand(self, lo, hi)
    }
}

pub trait RandOp<T>: Device<T> {
    fn rand(&self, x: &mut Matrix<T>, lo: T, hi: T);
}

pub fn rand_slice<T: SampleUniform + PartialOrd + Copy>(slice: &mut [T], lo: T, hi: T) {
    let mut rng = thread_rng();
    for value in slice {
        *value = rng.gen_range(lo..hi);
    }
}

impl<T: Float + SampleUniform> RandOp<T> for CPU {
    fn rand(&self, x: &mut Matrix<T>, lo: T, hi: T) {
        rand_slice(x, lo, hi)
    }
}

#[cfg(feature="opencl")]
impl<T: Float + SampleUniform> RandOp<T> for CLDevice {
    fn rand(&self, x: &mut Matrix<T>, lo: T, hi: T) {
        if self.unified_mem() {
            return rand_slice(x, lo, hi)
        }
        let mut data = vec![T::default(); x.len()];
        rand_slice(&mut data, lo, hi);
        cl_write(self, x, &data);
    }
}

#[cfg(feature="cuda")]
use custos::{CudaDevice, cuda::api::cu_write};

#[cfg(feature="cuda")]
impl<T: Float + SampleUniform> RandOp<T> for CudaDevice {
    fn rand(&self, x: &mut Matrix<T>, lo: T, hi: T) {
        let mut data = vec![T::default(); x.len()];
        rand_slice(&mut data, lo, hi);
        cu_write(x.ptr.2, &mut data).unwrap();
    }
}