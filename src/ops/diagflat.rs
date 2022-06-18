use custos::{
    cpu::{CPUCache, InternCPU},
    get_device,
    opencl::{InternCLDevice, KernelOptions},
    CDatatype, Matrix, Buffer,
};

use super::cl_to_cpu_s;

pub trait Diagflat<T> {
    fn diagflat(&self) -> Matrix<T>;
}

impl<T: CDatatype> Diagflat<T> for Matrix<T> {
    fn diagflat(&self) -> Matrix<T> {
        let device = get_device!(DiagflatOp, T).unwrap();
        device.diagflat(self)
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

impl<T: Default + Copy> DiagflatOp<T> for InternCPU {
    fn diagflat(&self, x: &Matrix<T>) -> Matrix<T> {
        assert!(x.dims().0 == 1 || x.dims().1 == 1);
        let size = x.size();

        let mut y = CPUCache::get::<T>(self.clone(), size*size);
        diagflat(x.as_slice(), y.as_mut_slice());
        (y, (size,size)).into()
    }
}

impl<T: CDatatype> DiagflatOp<T> for InternCLDevice {
    fn diagflat(&self, x: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_s(self, x, |device, x| device.diagflat(&x))
    }
}

pub fn cl_diagflat<T: CDatatype>(device: &InternCLDevice, x: &Matrix<T>) -> custos::Result<Buffer<T>> {
    let src = format!(
        r#"__kernel void diagflat(__global const {datatype}* input, const {datatype} co, __global {datatype}* output) {{
            size_t x = get_global_id(0);
            size_t y = get_global_id(1);
            
            int cols = (int) co;
            output[x * cols + x + y * cols * cols] = input[x + y*cols];
            
        }}"#, datatype = T::as_c_type_str()
    );

    KernelOptions::new(device, x, [x.cols(), x.rows(), 0], &src)?
        .add_arg(&T::from_usize(x.cols()))
        .with_output(x.cols() * x.cols() * x.rows())
        .run()
}