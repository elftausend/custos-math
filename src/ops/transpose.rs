#[cfg(feature = "cuda")]
use std::ptr::null_mut;

use crate::Matrix;
use custos::{cpu::CPU, get_device, CDatatype, Cache};

#[cfg(feature = "cuda")]
use custos::{
    cuda::api::cublas::{cublasDgeam, cublasOperation_t, cublasSgeam, CublasHandle},
    CUdeviceptr,
};

#[cfg(feature="opencl")]
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

impl<'a, T: CDatatype + CudaTranspose> Matrix<'a, T> {
    #[allow(non_snake_case)]
    pub fn T(&self) -> Matrix<'a, T> {
        get_device!(self.device(), TransposeOp<T>).transpose(self)
    }
}

pub trait TransposeOp<T> {
    fn transpose(&self, x: &Matrix<T>) -> Matrix<T>;
}

impl<T: Default + Copy> TransposeOp<T> for CPU {
    fn transpose(&self, x: &Matrix<T>) -> Matrix<T> {
        let mut out = Cache::get(self, x.len, x.node.idx);
        slice_transpose(x.rows(), x.cols(), x.as_slice(), out.as_mut_slice());
        (out, x.cols(), x.rows()).into()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> TransposeOp<T> for custos::CLDevice {
    fn transpose(&self, x: &Matrix<T>) -> Matrix<T> {
        Matrix {
            data: cl_transpose(self, x, x.rows(), x.cols()).unwrap(),
            dims: (x.cols(), x.rows()),
        }
    }
}

#[cfg(feature = "cuda")]
impl<T: CudaTranspose> TransposeOp<T> for custos::CudaDevice {
    fn transpose(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = Cache::get(self, x.len(), x.node.idx);
        T::transpose(&self.handle(), x.rows(), x.cols(), x.ptr.2, out.ptr.2).unwrap();
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
