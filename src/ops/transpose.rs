#[cfg(feature="cuda")]
use std::ptr::null_mut;

use custos::{
    cpu::CPU,
    get_device,
    CDatatype,
};
use crate::{cached, Matrix};

#[cfg(feature="opencl")]
use custos::{CLDevice, opencl::KernelOptions};

#[cfg(feature="cuda")]
use custos::{CUdeviceptr, cuda::{api::cublas::{CublasHandle, cublasSgeam, cublasOperation_t, cublasDgeam}, CudaCache}};

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

#[cfg(feature="opencl")]
pub fn cl_transpose<T: CDatatype>(
    device: CLDevice,
    x: &Matrix<T>,
) -> custos::Result<Matrix<T>> {
    let src = format!(
        "
        #define MODULO(x,N) (x % N)
        #define I0 {rows}
        #define I1 {cols}
        #define I_idx(i0,i1) ((size_t)(i0))*I1+(i1)
        #define I_idx_mod(i0,i1) MODULO( ((size_t)(i0)) ,I0)*I1+MODULO( (i1),I1)

        #define MODULO(x,N) (x % N)
        #define O0 {cols}
        #define O1 {rows}
        #define O_idx(o0,o1) ((size_t)(o0))*O1+(o1)
        #define O_idx_mod(o0,o1) MODULO( ((size_t)(o0)) ,O0)*O1+MODULO( (o1),O1)
        __kernel void transpose(__global const {datatype}* I, __global {datatype}* O) {{
            size_t gid = get_global_id(0);
            size_t gid_original = gid;size_t i1 = gid % I1;size_t i0 = gid / I1;gid = gid_original;
        
            O[O_idx(i1,i0)] = I[gid];
        }}
    
   ",
        rows = x.rows(),
        cols = x.cols(),
        datatype = T::as_c_type_str()
    );

    let gws = [x.size(), 0, 0];
    let buf = KernelOptions::new(&device, x, gws, &src)?
        .with_output(x.cols() * x.rows())
        .run();
    buf.map(|buf| (buf.unwrap(), (x.cols(), x.rows())).into())
}

impl<T: CDatatype + CudaTranspose> Matrix<T> {
    #[allow(non_snake_case)]
    pub fn T(&self) -> Matrix<T> {
        get_device!(TransposeOp<T>).unwrap().transpose(self)
    }
}

pub trait TransposeOp<T> {
    fn transpose(&self, x: &Matrix<T>) -> Matrix<T>;
}

impl<T: Default + Copy> TransposeOp<T> for CPU {
    fn transpose(&self, x: &Matrix<T>) -> Matrix<T> {
        let mut out = cached(self, (x.cols(), x.rows()));
        slice_transpose(x.rows(), x.cols(), x.as_slice(), out.as_mut_slice());
        out
    }
}

#[cfg(feature="opencl")]
impl<T: CDatatype> TransposeOp<T> for CLDevice {
    fn transpose(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_transpose(self.clone(), x).unwrap()
    }
}

#[cfg(feature="cuda")]
impl<T: CudaTranspose> TransposeOp<T> for custos::CudaDevice {
    fn transpose(&self, x: &Matrix<T>) -> Matrix<T> {
        let out = CudaCache::get(self, x.len());
        T::transpose(
            &self.handle(), x.rows(), x.cols(), x.ptr.2, out.ptr.2
        ).unwrap();
        (out, x.cols(), x.rows()).into()
    }
}

pub trait CudaTranspose {
    #[cfg(feature="cuda")]
    fn transpose(handle: &CublasHandle, m: usize, n: usize, a: CUdeviceptr, c: CUdeviceptr) -> custos::Result<()>;
}

impl CudaTranspose for f32 {
    #[cfg(feature="cuda")]
    fn transpose(handle: &CublasHandle, m: usize, n: usize, a: CUdeviceptr, c: CUdeviceptr) -> custos::Result<()> {
        unsafe {
            // TODO: better casting than: usize as i32
            cublasSgeam(
                handle.0, cublasOperation_t::CUBLAS_OP_T, 
                cublasOperation_t::CUBLAS_OP_N, m as i32, n as i32, 
                &1f32 as *const f32, a as *const CUdeviceptr as *const f32, n as i32, 
                &0f32 as *const f32, null_mut(), m as i32, 
                c as *mut CUdeviceptr as *mut f32, m as i32
            ).to_result()?;
        }
        Ok(())
    }
}

impl CudaTranspose for f64 {
    #[cfg(feature="cuda")]
    fn transpose(handle: &CublasHandle, m: usize, n: usize, a: CUdeviceptr, c: CUdeviceptr) -> custos::Result<()> {
        unsafe {
            // TODO: better casting than: usize as i32
            cublasDgeam(
                handle.0, cublasOperation_t::CUBLAS_OP_T, 
                cublasOperation_t::CUBLAS_OP_N, m as i32, n as i32, 
                &1f64 as *const f64, a as *const CUdeviceptr as *const f64, n as i32, 
                &0f64 as *const f64, null_mut(), m as i32, 
                c as *mut CUdeviceptr as *mut f64, m as i32
            ).to_result()?;
        }
        Ok(())
    }
}