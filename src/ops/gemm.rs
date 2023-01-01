use custos::{impl_stack, Device, Dim2, GenericBlas, MainMemory, Shape, CPU};

#[cfg(feature="cpu")]
use custos::cache::Cache;

#[cfg(feature = "stack")]
use custos::Stack;

#[cfg(feature = "opencl")]
use custos::CDatatype;

#[cfg(feature = "opencl")]
use crate::cl_gemm;
#[cfg(feature = "opencl")]
use custos::OpenCL;

use crate::Matrix;

impl<'a, T, D: Device> Matrix<'a, T, D> {
    /// Matrix multiplication. Uses current global device.
    /// # Example
    /// ```
    /// use custos::CPU;
    /// use custos_math::Matrix;
    ///
    /// let device = CPU::new();
    ///
    /// let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.,]));
    /// let b = Matrix::from((&device, (3, 2), [6., 5., 4., 3., 2., 1.,]));
    ///
    /// let c = a.gemm(&b);
    /// println!("c: {c:?}");
    ///
    /// assert_eq!(c.read(), vec![20., 14., 56., 41.,]);
    /// ```
    #[inline]
    pub fn gemm<'b>(&'a self, rhs: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D: Gemm<T, D>,
    {
        self.device().gemm(self, rhs)
    }
}

impl<'a, T, D: Device, const M: usize, const K: usize> Matrix<'a, T, D, Dim2<M, K>> {
    /// Matrix multiplication. Uses current global device.
    /// # Example
    /// ```
    /// use custos::CPU;
    /// use custos_math::Matrix;
    ///
    /// let device = CPU::new();
    ///
    /// let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.,]));
    /// let b = Matrix::from((&device, (3, 2), [6., 5., 4., 3., 2., 1.,]));
    ///
    /// let c = a.gemm(&b);
    /// println!("c: {c:?}");
    ///
    /// assert_eq!(c.read(), vec![20., 14., 56., 41.,]);
    /// ```
    #[inline]
    pub fn gemm<'b, const N: usize>(
        &'a self,
        rhs: &Matrix<'a, T, D, Dim2<K, N>>,
    ) -> Matrix<'a, T, D, Dim2<M, N>>
    where
        D: Gemm<T, D, Dim2<M, K>, Dim2<K, N>, Dim2<M, N>>,
    {
        self.device().gemm(self, rhs)
    }
}

/// Matrix multiplication. Uses provided device.
/// # Example
/// ```
/// use custos::{CPU, Read};
/// use custos_math::{Matrix, Gemm};
///
/// let device = CPU::new();
///
/// let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.,]));
/// let b = Matrix::from((&device, (3, 2), [6., 5., 4., 3., 2., 1.,]));
///
/// let c: Matrix = device.gemm(&a, &b);
///
/// assert_eq!(c.read(), vec![20., 14., 56., 41.,]);
/// ```
pub trait Gemm<T, D: Device = Self, LS: Shape = (), RS: Shape = (), OS: Shape = ()>:
    Device
{
    fn gemm(&self, lhs: &Matrix<T, D, LS>, rhs: &Matrix<T, D, RS>) -> Matrix<T, Self, OS>;
}


#[cfg(not(feature="no-std"))]
#[cfg(feature="cpu")]
impl<T, D, LS, RS, OS> Gemm<T, D, LS, RS, OS> for CPU
where
    T: GenericBlas + Default + Copy,
    D: MainMemory,
    LS: Shape,
    RS: Shape,
    OS: Shape,
{
    #[inline]
    fn gemm(&self, lhs: &Matrix<T, D, LS>, rhs: &Matrix<T, D, RS>) -> Matrix<T, Self, OS> {
        assert!(lhs.dims().1 == rhs.dims().0);
        let (m, k) = lhs.dims();
        let n = rhs.cols();

        let mut out = self.retrieve(m * n, (lhs.node.idx, rhs.node.idx));
        T::gemm(m, n, k, lhs, rhs, &mut out);
        (out, m, n).into()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> Gemm<T> for OpenCL {
    fn gemm(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        assert!(lhs.dims().1 == rhs.dims().0);
        //crate::opencl::ops::ocl_gemm1(self.clone(), rhs, lhs).unwrap()
        let buf = cl_gemm(self, rhs.cols(), rhs.rows(), lhs.rows(), rhs, lhs).unwrap();
        (buf, lhs.rows(), rhs.cols()).into()
    }
}

#[cfg(feature = "cuda")]
impl<T: GenericBlas> Gemm<T> for custos::CUDA {
    fn gemm(
        &self,
        lhs: &Matrix<T, custos::CUDA>,
        rhs: &Matrix<T, custos::CUDA>,
    ) -> Matrix<T, custos::CUDA> {
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
            lhs.as_buf().ptr.ptr,
            rhs.as_buf().ptr.ptr,
            out.ptr.ptr,
        )
        .unwrap();
        (out, lhs.rows(), rhs.cols()).into()
    }
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "stack")]
    #[test]
    fn test_stack_impl() {
        use custos::{Buffer, Dim1, Dim2, Stack};

        use crate::Matrix;

        let data = Buffer::from((Stack, &[3., 1., 5.]));
        let lhs = Matrix { data, dims: (1, 3) };

        /*let data = Buffer::<_, _, Dim2<3, 1>>::from((Stack, &[3., 1., 5.]));
        let rhs = Matrix { data, dims: (3, 1) };

        let out: Matrix<f64, Stack, Dim1<1>> = lhs.gemm(&rhs);*/
    }
}
