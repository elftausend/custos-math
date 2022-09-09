use custos::{cache::Cache, GenericBlas, CPU};

#[cfg(feature = "opencl")]
use custos::CDatatype;

#[cfg(feature = "opencl")]
use crate::cl_gemm;
#[cfg(feature = "opencl")]
use custos::CLDevice;

use crate::Matrix;

/// Matrix multiplication. Uses provided device.
/// # Example
/// ```
/// use custos::{CPU, VecRead};
/// use custos_math::{Matrix, Gemm};
///
/// let device = CPU::new();
///
/// let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.,]));
/// let b = Matrix::from((&device, (3, 2), [6., 5., 4., 3., 2., 1.,]));
///
/// let c = device.gemm(&a, &b);
///
/// assert_eq!(device.read(c.as_buf()), vec![20., 14., 56., 41.,]);
/// ```
pub trait Gemm<T> {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
}

impl<T: GenericBlas + Default + Copy> Gemm<T> for CPU {
    #[inline]
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(lhs.dims().1 == rhs.dims().0);
        let m = lhs.dims().0;
        let k = lhs.dims().1;
        let n = rhs.dims().1;

        let mut c = Cache::get(self, m * n, [lhs.node.idx, rhs.node.idx]);
        T::gemm(m, n, k, lhs, rhs, &mut c);
        (c, (m, n)).into()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> Gemm<T> for CLDevice {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(lhs.dims().1 == rhs.dims().0);
        //crate::opencl::ops::ocl_gemm1(self.clone(), rhs, lhs).unwrap()
        let buf = cl_gemm(self, rhs.cols(), rhs.rows(), lhs.rows(), rhs, lhs).unwrap();
        (buf, (lhs.rows(), rhs.cols())).into()
    }
}

#[cfg(feature = "cuda")]
impl<T: GenericBlas> Gemm<T> for custos::CudaDevice {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        use custos::CacheBuf;
        assert!(
            lhs.cols() == rhs.rows(),
            "wrong dims for matrix multiplication"
        );
        let out = self.cached(lhs.rows() * rhs.cols());
        T::cugemm(
            self.handle(),
            lhs.rows(),
            rhs.cols(),
            lhs.cols(),
            lhs.as_buf().ptr.2,
            rhs.as_buf().ptr.2,
            out.ptr.2,
        )
        .unwrap();
        (out, lhs.rows(), rhs.cols()).into()
    }
}
