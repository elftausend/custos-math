#[cfg(feature = "cuda")]
use crate::cu_to_cpu_s;
use crate::Matrix;
#[cfg(feature = "cuda")]
use custos::CudaDevice;
use custos::{
    cpu::{CPUCache, CPU},
    get_device, CDatatype,
};

#[cfg(feature = "opencl")]
use super::cl_to_cpu_s;
#[cfg(feature = "opencl")]
use custos::{Buffer, CLDevice};

impl<T: CDatatype> Matrix<T> {
    pub fn diagflat(&self) -> Matrix<T> {
        get_device!(DiagflatOp<T>).unwrap().diagflat(self)
    }
}

pub fn diagflat<T: Copy>(a: &[T], b: &mut [T]) {
    for (row, x) in a.iter().enumerate() {
        b[row * a.len() + row] = *x;
    }
}

pub trait DiagflatOp<T> {
    fn diagflat(&self, x: &Matrix<T>) -> Matrix<T>;
}

impl<T: Default + Copy> DiagflatOp<T> for CPU {
    fn diagflat(&self, x: &Matrix<T>) -> Matrix<T> {
        assert!(x.dims().0 == 1 || x.dims().1 == 1);
        let size = x.size();

        let mut y = CPUCache::get::<T>(self, size * size);
        diagflat(x.as_slice(), y.as_mut_slice());
        (y, (size, size)).into()
    }
}

#[cfg(feature = "cuda")]
impl<T: Copy + Default> DiagflatOp<T> for CudaDevice {
    fn diagflat(&self, x: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_s(self, x, |cpu, x| cpu.diagflat(&x))
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> DiagflatOp<T> for CLDevice {
    fn diagflat(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_s(self, x, |device, x| device.diagflat(&x))
    }
}

#[cfg(feature = "opencl")]
pub fn cl_diagflat<T: CDatatype>(device: &CLDevice, x: &Matrix<T>) -> custos::Result<Buffer<T>> {
    use custos::opencl::{CLCache, enqueue_kernel};

    let src = format!(
        r#"__kernel void diagflat(__global const {datatype}* input, const int cols, __global {datatype}* output) {{
            size_t x = get_global_id(0);
            size_t y = get_global_id(1);
        
            output[x * cols + x + y * cols * cols] = input[x + y*cols];
            
        }}"#,
        datatype = T::as_c_type_str()
    );

    let out = CLCache::get::<T>(device, x.cols() * x.cols() * x.rows());
    enqueue_kernel(device, &src, [x.cols(), x.rows(), 0], None, &[x, &(x.cols() as i32), &out])?;
    Ok(out)
}
