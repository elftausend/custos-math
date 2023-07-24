use crate::Matrix;
#[cfg(feature = "opencl")]
use custos::OpenCL;
use custos::{impl_stack, number::Float, Buffer, Device, MainMemory, Shape, CPU};
//use rand::{thread_rng, Rng, distributions::uniform::SampleUniform};

#[cfg(feature = "stack")]
use custos::Stack;

#[cfg(feature = "opencl")]
use crate::opencl::cl_write;

pub trait RandBuf<T> {
    fn rand(&mut self, lo: T, hi: T);
}
impl<T, S: Shape, D: RandOp<T, S>> RandBuf<T> for Buffer<'_, T, D, S> {
    #[inline]
    fn rand(&mut self, lo: T, hi: T) {
        self.device().rand(self, lo, hi)
    }
}

impl<'a, T, S: Shape, D: RandOp<T, S>> Matrix<'a, T, D, S> {
    #[inline]
    pub fn rand(&mut self, lo: T, hi: T) {
        self.as_buf_mut().rand(lo, hi);
    }
}

pub trait RandOp<T, S: Shape = (), D: Device = Self>: Device {
    fn rand(&self, x: &mut Buffer<T, D, S>, lo: T, hi: T);
}

pub fn rand_slice<T: PartialOrd + Copy + Float>(slice: &mut [T], lo: T, hi: T) {
    let rng = fastrand::Rng::new();
    for value in slice {
        *value = T::as_generic(rng.f64()) * (hi - (lo)) + (lo);
    }
}

#[impl_stack]
impl<T: Float, D: MainMemory, S: Shape> RandOp<T, S, D> for CPU {
    #[inline]
    fn rand(&self, x: &mut Buffer<T, D, S>, lo: T, hi: T) {
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
use custos::CUDA;

#[cfg(feature = "cuda")]
impl<T: Float> RandOp<T> for CUDA {
    fn rand(&self, x: &mut Buffer<T, CUDA>, lo: T, hi: T) {
        let mut data = vec![T::default(); x.len()];
        rand_slice(&mut data, lo, hi);
        x.write(&data);
        //cu_write(x.ptr.ptr, &data).unwrap();
    }
}
