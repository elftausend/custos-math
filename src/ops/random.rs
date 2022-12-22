use crate::Matrix;
#[cfg(feature = "opencl")]
use custos::OpenCL;
use custos::{number::Float, Buffer, Device, MainMemory, CPU};
//use rand::{thread_rng, Rng, distributions::uniform::SampleUniform};

#[cfg(feature = "opencl")]
use crate::opencl::cl_write;

pub trait RandBuf<T> {
    fn rand(&mut self, lo: T, hi: T);
}
impl<T: Float, D: RandOp<T>> RandBuf<T> for Buffer<'_, T, D> {
    fn rand(&mut self, lo: T, hi: T) {
        self.device().rand(self, lo, hi)
    }
}

impl<'a, T: Float> Matrix<'a, T> {
    pub fn rand(&mut self, lo: T, hi: T) {
        self.as_mut_buf().rand(lo, hi);
    }
}

pub trait RandOp<T, D: Device = Self>: Device {
    fn rand(&self, x: &mut Buffer<T, D>, lo: T, hi: T);
}

pub fn rand_slice<T: PartialOrd + Copy + Float>(slice: &mut [T], lo: T, hi: T) {
    let rng = fastrand::Rng::new();
    for value in slice {
        *value = T::as_generic(rng.f64()) * (hi - (lo)) + (lo);
    }
}

impl<T: Float, D: MainMemory> RandOp<T, D> for CPU {
    fn rand(&self, x: &mut Buffer<T, D>, lo: T, hi: T) {
        rand_slice(x, lo, hi)
    }
}

#[cfg(feature = "opencl")]
impl<T: Float> RandOp<T> for OpenCL {
    fn rand(&self, x: &mut Buffer<T, OpenCL>, lo: T, hi: T) {
        #[cfg(unified_cl)]
        rand_slice(x, lo, hi);

        #[cfg(not(unified_cl))]
        {
            let mut data = vec![T::default(); x.len()];
            rand_slice(&mut data, lo, hi);
            cl_write(self, x, &data)
        };
    }
}

#[cfg(feature = "cuda")]
use custos::{cuda::api::cu_write, CUDA};

#[cfg(feature = "cuda")]
impl<T: Float> RandOp<T> for CUDA {
    fn rand(&self, x: &mut Buffer<T, CUDA>, lo: T, hi: T) {
        let mut data = vec![T::default(); x.len()];
        rand_slice(&mut data, lo, hi);
        cu_write(x.ptr.ptr, &data).unwrap();
    }
}
