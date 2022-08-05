use custos::{CDatatype, GenericBlas};
use custos_math::{Matrix, CudaTranspose};

pub struct Owns<'a, T> {
    other: Matrix<'a, T>,
    buf: Option<&'a Matrix<'a, T>>
}


impl<'a, T: CudaTranspose + CDatatype + GenericBlas> Owns<'a, T> {
    pub fn test(&mut self, inputs: &'a Matrix<T>) {
        let mat: &Matrix<T> = self.buf.unwrap();
        /*let device = mat.device();
        let cpu: &'a CPU = unsafe {&*(device.device as *mut CPU)};*/
        
        /*let trans: Matrix<'a, T> = cpu.transpose(mat);*/
        //self.other = mat.t.gemm(inputs);
        self.other = mat.T().gemm(inputs);
    }  
}
