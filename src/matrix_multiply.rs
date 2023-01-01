#[rustfmt::skip]
pub trait MatrixMultiply where Self: Sized {
    fn gemm(m: usize, k: usize, n: usize,
        a: &[Self], rsa: usize, csa: usize,
        b: &[Self], rsb: usize, csb: usize,
        c: &mut [Self], rsc: usize, csc: usize);
}

#[cfg(feature = "matrixmultiply")]
mod implements {
    use super::MatrixMultiply;

    #[rustfmt::skip]
    impl MatrixMultiply for f32 {
        #[inline]
        fn gemm(m: usize, k: usize, n: usize,
            a: &[Self], rsa: usize, csa: usize,
            b: &[Self], rsb: usize, csb: usize,
            c: &mut [Self], rsc: usize, csc: usize) {
            
                unsafe {
                    matrixmultiply::sgemm(m, k, n, 1., a.as_ptr(), rsa as isize, csa as isize, b.as_ptr(), rsb as isize, csb as isize, 1., c.as_mut_ptr(), rsc as isize, csc as isize);
                }
                
        }
    }
    
    #[rustfmt::skip]
    impl MatrixMultiply for f64 {
        #[inline]
        fn gemm(m: usize, k: usize, n: usize,
            a: &[Self], rsa: usize, csa: usize,
            b: &[Self], rsb: usize, csb: usize,
            c: &mut [Self], rsc: usize, csc: usize) 
            
            {
            unsafe {
                matrixmultiply::dgemm(m, k, n, 1., a.as_ptr(), rsa as isize, csa as isize, b.as_ptr(), rsb as isize, csb as isize, 1., c.as_mut_ptr(), rsc as isize, csc as isize);
            }           
        }
    }
    
}
