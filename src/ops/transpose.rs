use custos::{Matrix, cpu::{InternCPU, CPUCache}, number::Number};


pub trait Transpose<T> {
    #[allow(non_snake_case)]
    fn T(&self, x: Matrix<T>) -> Matrix<T>;
}

impl <T: Number>Transpose<T> for InternCPU {
    #[allow(non_snake_case)]
    fn T(&self, x: Matrix<T>) -> Matrix<T> {
        
        todo!()
    }
}