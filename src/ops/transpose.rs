use custos::{Matrix, cpu::{InternCPU, CPUCache}, opencl::{GenericOCL, InternCLDevice, KernelOptions}, Error, get_device};

pub fn slice_transpose<T: Copy>(rows: usize, cols: usize, a: &[T], b: &mut [T]) {
    for i in 0..rows {
        let index = i*cols;
        let row = &a[index..index+cols];
        
        for (index, row) in row.iter().enumerate() {
            let idx = rows*index+i;
            b[idx] = *row;
        } 
    }
}

pub fn cl_transpose<T: GenericOCL>(device: InternCLDevice, x: Matrix<T>) -> Result<Matrix<T>, Error>{
    let src = format!("
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
    
   ", rows=x.rows(), cols=x.cols(), datatype=T::as_ocl_type_str());

    let gws = [x.size(), 0, 0];
    KernelOptions::new(&device, x, gws, &src)
        .with_output((x.cols(), x.rows()))
        .run()
}

pub trait Transpose<T> {
    #[allow(non_snake_case)]
    fn T(self) -> Matrix<T>;
}

impl <T: GenericOCL>Transpose<T> for Matrix<T> {
    #[allow(non_snake_case)]
    fn T(self) -> Matrix<T> {
        let device = get_device!(TransposeOp, T).unwrap();
        device.transpose(self)
    }
}

pub trait TransposeOp<T> {
    fn transpose(&self, x: Matrix<T>) -> Matrix<T>;
}

impl <T: Default+Copy>TransposeOp<T> for InternCPU {
    fn transpose(&self, x: Matrix<T>) -> Matrix<T> {
        let mut y = CPUCache::get::<T>(self.clone(), (x.cols(), x.rows()));
        slice_transpose(x.rows(), x.cols(), x.as_cpu_slice(), y.as_cpu_slice_mut());
        y
    }
}

impl <T: GenericOCL>TransposeOp<T> for InternCLDevice {
    fn transpose(&self, x: Matrix<T>) -> Matrix<T> {
        cl_transpose(self.clone(), x).unwrap()
    }
}