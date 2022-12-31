use custos::{impl_stack, number::Number, Device, MainMemory, Shape, CPU};

use crate::{ew_op, Matrix};

#[cfg(feature = "stack")]
use custos::Stack;

#[cfg(any(feature = "cuda", feature = "opencl"))]
use custos::CDatatype;

#[cfg(feature = "opencl")]
use crate::cl_tew;
#[cfg(feature = "opencl")]
use custos::OpenCL;

#[cfg(feature = "cuda")]
use crate::cu_ew;

#[cfg_attr(feature = "safe", doc = "```ignore")]
/// Element-wise +, -, *, / operations for matrices.
///
/// # Examples
/// ```
/// use custos::CPU;
/// use custos_math::Matrix;
///
/// let device = CPU::new();
/// let a = Matrix::from((&device, (2, 3), [2, 4, 6, 8, 10, 12]));
/// let b = Matrix::from((&device, (2, 3), [12, 4, 3, 1, -5, -3]));
///
/// let c = &a + &b;
/// assert_eq!(c.read(), vec![14, 8, 9, 9, 5, 9]);
///
/// use custos_math::BaseOps;
/// let sub = device.sub(&a, &b);
/// assert_eq!(sub.read(), vec![-10, 0, 3, 7, 15, 15]);
/// ```
pub trait BaseOps<T, D: Device = Self, S: Shape = ()>: Device {
    /// Element-wise addition
    /// # Example
    /// ```
    /// use custos::CPU;
    /// use custos_math::Matrix;
    ///
    /// let device = CPU::new();
    /// let a = Matrix::from((&device, 2, 3, [2, 4, 6, 8, 10, 12]));
    /// let b = Matrix::from((&device, 2, 3, [12, 4, 3, 1, -5, -3]));
    ///
    /// let c = a + b;
    /// assert_eq!(c.read(), vec![14, 8, 9, 9, 5, 9]);
    /// ```
    fn add(&self, lhs: &Matrix<T, D, S>, rhs: &Matrix<T, D, S>) -> Matrix<T, Self, S>;

    /// Element-wise subtraction
    /// # Example
    /// ```
    /// use custos::CPU;
    /// use custos_math::{Matrix, BaseOps};
    ///
    /// let device = CPU::new();
    /// let a = Matrix::from((&device, 2, 3, [2, 4, 6, 8, 10, 12]));
    /// let b = Matrix::from((&device, 2, 3, [12, 4, 3, 1, -5, -3]));
    ///
    /// let sub = device.sub(&a, &b);
    /// assert_eq!(sub.read(), vec![-10, 0, 3, 7, 15, 15]);
    /// ```
    fn sub(&self, lhs: &Matrix<T, D, S>, rhs: &Matrix<T, D, S>) -> Matrix<T, Self, S>;

    /// Element-wise multiplication
    /// # Example
    /// ```
    /// use custos::CPU;
    /// use custos_math::{Matrix, BaseOps};
    ///
    /// let device = CPU::new();
    /// let a = Matrix::from((&device, 2, 3, [2, 4, 6, 8, 10, 12]));
    /// let b = Matrix::from((&device, 2, 3, [12, 4, 3, 1, -5, -3]));
    ///
    /// let mul = a * b;
    /// assert_eq!(mul.read(), vec![24, 16, 18, 8, -50, -36]);
    /// ```
    fn mul(&self, lhs: &Matrix<T, D, S>, rhs: &Matrix<T, D, S>) -> Matrix<T, Self, S>;

    /// Element-wise division
    /// # Example
    /// ```
    /// use custos::CPU;
    /// use custos_math::{Matrix, BaseOps};
    ///
    /// let device = CPU::new();
    /// let a = Matrix::from((&device, 2, 3, [2, 4, 6, 8, 10, 12]));
    /// let b = Matrix::from((&device, 2, 3, [12, 4, 3, 1, -5, -3]));
    ///
    /// let div = device.div(&a, &b);
    /// assert_eq!(div.read(), vec![0, 1, 2, 8, -2, -4]);
    /// ```
    fn div(&self, lhs: &Matrix<T, D, S>, rhs: &Matrix<T, D, S>) -> Matrix<T, Self, S>;
}

#[impl_stack]
impl<T, S, D> BaseOps<T, D, S> for CPU
where
    T: Number,
    S: Shape,
    D: MainMemory,
{
    fn add(&self, lhs: &Matrix<T, D, S>, rhs: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        ew_op(self, lhs, rhs, |x, y| x + y)
    }

    fn sub(&self, lhs: &Matrix<T, D, S>, rhs: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        ew_op(self, lhs, rhs, |x, y| x - y)
    }

    fn mul(&self, lhs: &Matrix<T, D, S>, rhs: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        ew_op(self, lhs, rhs, |x, y| x * y)
    }

    fn div(&self, lhs: &Matrix<T, D, S>, rhs: &Matrix<T, D, S>) -> Matrix<T, Self, S> {
        ew_op(self, lhs, rhs, |x, y| x / y)
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> BaseOps<T> for OpenCL {
    fn add(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        let buf = cl_tew(self, lhs, rhs, "+").unwrap();
        (buf, lhs.dims()).into()
    }

    fn sub(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        let buf = cl_tew(self, lhs, rhs, "-").unwrap();
        (buf, lhs.dims()).into()
    }

    fn mul(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        let buf = cl_tew(self, lhs, rhs, "*").unwrap();
        (buf, lhs.dims()).into()
    }

    fn div(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        let buf = cl_tew(self, lhs, rhs, "/").unwrap();
        (buf, lhs.dims()).into()
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> BaseOps<T> for custos::CUDA {
    fn add(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        let buf = cu_ew(self, lhs, rhs, "+").unwrap();
        (buf, lhs.dims()).into()
    }

    fn sub(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        let buf = cu_ew(self, lhs, rhs, "-").unwrap();
        (buf, lhs.dims()).into()
    }

    fn mul(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        let buf = cu_ew(self, lhs, rhs, "*").unwrap();
        (buf, lhs.dims()).into()
    }

    fn div(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        let buf = cu_ew(self, lhs, rhs, "/").unwrap();
        (buf, lhs.dims()).into()
    }

    /*fn clear(&self, buf: &mut crate::Buffer<T>) {
        cu_clear(self, buf).unwrap();
    }*/
}
