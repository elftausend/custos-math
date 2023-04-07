#[cfg(feature = "cuda")]
use std::ptr::null_mut;

use crate::Matrix;
use custos::{CDatatype, Device, MainMemory, Shape};

#[cfg(feature = "cpu")]
use custos::CPU;

#[cfg(feature = "cuda")]
use custos::{
    cuda::api::{cublas::{cublasDgeam, cublasOperation_t, cublasSgeam, CublasHandle}, CUdeviceptr},
};

#[cfg(feature = "opencl")]
use crate::cl_transpose;

pub fn slice_transpose<T: Copy>(rows: usize, cols: usize, a: &[T], b: &mut [T]) {
    for i in 0..rows {
        let index = i * cols;
        let row = &a[index..index + cols];

        for (index, row) in row.iter().enumerate() {
            let idx = rows * index + i;
            b[idx] = *row;
        }
    }
}

impl<'a, T, IS: Shape, D: Device> Matrix<'a, T, D, IS> {
    #[allow(non_snake_case)]
    pub fn T<OS: Shape>(&self) -> Matrix<'a, T, D, OS>
    where
        D: TransposeOp<T, IS, OS>,
    {
        self.device().transpose(self)
    }
}

pub trait TransposeOp<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn transpose(&self, x: &Matrix<T, D, IS>) -> Matrix<T, Self, OS>;
}

#[cfg(feature = "cpu")]
impl<T: Default + Copy, D: MainMemory, IS: Shape, OS: Shape> TransposeOp<T, IS, OS, D> for CPU {
    fn transpose(&self, x: &Matrix<T, D, IS>) -> Matrix<T, Self, OS> {
        let mut out = self.retrieve(x.len(), x.as_buf());
        slice_transpose(x.rows(), x.cols(), x.as_slice(), out.as_mut_slice());
        (out, x.cols(), x.rows()).into()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> TransposeOp<T> for custos::OpenCL {
    fn transpose(&self, x: &Matrix<T, custos::OpenCL>) -> Matrix<T, custos::OpenCL> {
        Matrix {
            data: cl_transpose(self, x, x.rows(), x.cols()).unwrap(),
            dims: (x.cols(), x.rows()),
        }
    }
}

#[cfg(feature = "cuda")]
impl<T: CudaTranspose> TransposeOp<T> for custos::CUDA {
    fn transpose(&self, x: &Matrix<T, custos::CUDA>) -> Matrix<T, custos::CUDA> {
        let out = self.retrieve(x.len(), x.as_buf());
        T::transpose(&self.cublas_handle(), x.rows(), x.cols(), x.ptr.ptr, out.ptr.ptr).unwrap();
        (out, x.cols(), x.rows()).into()
    }
}

pub trait CudaTranspose {
    #[cfg(feature = "cuda")]
    fn transpose(
        handle: &CublasHandle,
        m: usize,
        n: usize,
        a: CUdeviceptr,
        c: CUdeviceptr,
    ) -> custos::Result<()>;
}

impl CudaTranspose for f32 {
    #[cfg(feature = "cuda")]
    fn transpose(
        handle: &CublasHandle,
        m: usize,
        n: usize,
        a: CUdeviceptr,
        c: CUdeviceptr,
    ) -> custos::Result<()> {
        unsafe {
            // TODO: better casting than: usize as i32
            cublasSgeam(
                handle.0,
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                m as i32,
                n as i32,
                &1f32 as *const f32,
                a as *const CUdeviceptr as *const f32,
                n as i32,
                &0f32 as *const f32,
                null_mut(),
                m as i32,
                c as *mut CUdeviceptr as *mut f32,
                m as i32,
            )
            .to_result()?;
        }
        Ok(())
    }
}

impl CudaTranspose for f64 {
    #[cfg(feature = "cuda")]
    fn transpose(
        handle: &CublasHandle,
        m: usize,
        n: usize,
        a: CUdeviceptr,
        c: CUdeviceptr,
    ) -> custos::Result<()> {
        unsafe {
            // TODO: better casting than: usize as i32
            cublasDgeam(
                handle.0,
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                m as i32,
                n as i32,
                &1f64 as *const f64,
                a as *const CUdeviceptr as *const f64,
                n as i32,
                &0f64 as *const f64,
                null_mut(),
                m as i32,
                c as *mut CUdeviceptr as *mut f64,
                m as i32,
            )
            .to_result()?;
        }
        Ok(())
    }
}
