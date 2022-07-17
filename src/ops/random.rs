use custos::{
    get_device, number::Float, Device, CPU, Buffer,
};
use crate::Matrix;
#[cfg(feature="opencl")]
use custos::CLDevice;
//use rand::{thread_rng, Rng, distributions::uniform::SampleUniform};

#[cfg(feature="opencl")]
use crate::opencl::cl_write;

pub trait RandBuf<T> {
    fn rand(&mut self, lo: T, hi: T);
}
impl<T: Float> RandBuf<T> for Buffer<T> {
    fn rand(&mut self, lo: T, hi: T) {
        let device = get_device!(RandOp<T>).unwrap();
        device.rand(self, lo, hi)
    }
}

impl<T: Float> Matrix<T> {
    pub fn rand(&mut self, lo: T, hi: T) {
        self.as_mut_buf().rand(lo, hi);
    }
}

pub trait RandOp<T>: Device<T> {
    fn rand(&self, x: &mut Buffer<T>, lo: T, hi: T);
}

pub fn rand_slice<T: PartialOrd + Copy + Float>(slice: &mut [T], lo: T, hi: T) {
    let rng = fastrand::Rng::new();
    for value in slice {
        *value = T::as_generic(rng.f64()) * (hi-(lo))+(lo);
    }
}

impl<T: Float > RandOp<T> for CPU {
    fn rand(&self, x: &mut Buffer<T>, lo: T, hi: T) {
        rand_slice(x, lo, hi)
    }
}

#[cfg(feature="opencl")]
impl<T: Float > RandOp<T> for CLDevice {
    fn rand(&self, x: &mut Buffer<T>, lo: T, hi: T) {
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
impl<T: Float > RandOp<T> for CudaDevice {
    fn rand(&self, x: &mut Buffer<T>, lo: T, hi: T) {
        let mut data = vec![T::default(); x.len()];
        rand_slice(&mut data, lo, hi);
        cu_write(x.ptr.2, &data).unwrap();
    }
}