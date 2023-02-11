use custos::{impl_stack, number::Number, Alloc, Buffer, Device, MainMemory, Shape, CPU};

#[cfg(feature = "stack")]
use custos::Stack;

#[cfg(any(feature = "cuda", feature = "opencl"))]
use custos::CDatatype;

#[cfg(feature = "opencl")]
use crate::cl_tew_self;
#[cfg(feature = "opencl")]
use custos::OpenCL;

#[cfg(feature = "cuda")]
use crate::cu_ew_self;
use crate::{assign_to_lhs, element_wise_op_mut, Matrix};

/// Assignment operations
/// # Examples
#[cfg_attr(feature = "cpu", doc = "```")]
#[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
/// use custos::{CPU, Read};
/// use custos_math::{Matrix, AssignOps};
///
/// let device = CPU::new();
/// let mut lhs = Matrix::from((&device, 2, 2, [3, 5, 4, 1]));
/// let rhs = Matrix::from((&device, 2, 2, [1, 8, 6, 2]));
///
/// device.add_assign(&mut lhs, &rhs);
/// assert_eq!(vec![4, 13, 10, 3], lhs.read());
///
/// device.sub_assign(&mut lhs, &rhs);
/// assert_eq!(vec![3, 5, 4, 1], lhs.read());
/// ```
pub trait AssignOps<T, S: Shape = (), D: Device = Self>: Device {
    /// Add assign
    /// # Examples
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Read};
    /// use custos_math::{Matrix, AssignOps};
    ///
    /// let device = CPU::new();
    /// let mut lhs = Matrix::from((&device, 2, 2, [3, 5, 4, 1]));
    /// let rhs = Matrix::from((&device, 2, 2, [1, 8, 6, 2]));
    ///
    /// device.add_assign(&mut lhs, &rhs);
    /// assert_eq!(vec![4, 13, 10, 3], lhs.read());
    ///
    /// device.sub_assign(&mut lhs, &rhs);
    /// assert_eq!(vec![3, 5, 4, 1], lhs.read());
    /// ```
    fn add_assign(&self, lhs: &mut Buffer<T, Self, S>, rhs: &Buffer<T, D, S>);
    fn sub_assign(&self, lhs: &mut Buffer<T, Self, S>, rhs: &Buffer<T, D, S>);
    fn mul_assign(&self, lhs: &mut Buffer<T, Self, S>, rhs: &Buffer<T, D, S>);
}

pub fn ew_op<'a, T, F, D, S, Host>(
    device: &'a Host,
    lhs: &Matrix<T, D, S>,
    rhs: &Matrix<T, D, S>,
    f: F,
) -> Matrix<'a, T, Host, S>
where
    T: Copy + Default,
    F: Fn(T, T) -> T,
    D: MainMemory,
    S: Shape,
    Host: for<'b> Alloc<'b, T, S> + MainMemory,
{
    let mut out = device.retrieve(lhs.size(), (lhs.node.idx, rhs.node.idx));
    element_wise_op_mut(lhs, rhs, &mut out, f);
    (out, lhs.dims()).into()
}

#[impl_stack]
impl<T: Number, D: MainMemory, S: Shape> AssignOps<T, S, D> for CPU {
    #[inline]
    fn add_assign(&self, lhs: &mut Buffer<T, Self, S>, rhs: &Buffer<T, D, S>) {
        assign_to_lhs(lhs, rhs, |x, y| *x += y)
    }

    #[inline]
    fn sub_assign(&self, lhs: &mut Buffer<T, Self, S>, rhs: &Buffer<T, D, S>) {
        assign_to_lhs(lhs, rhs, |x, y| *x -= y)
    }

    #[inline]
    fn mul_assign(&self, lhs: &mut Buffer<T, Self, S>, rhs: &Buffer<T, D, S>) {
        assign_to_lhs(lhs, rhs, |x, y| *x *= y)
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> AssignOps<T> for OpenCL {
    #[inline]
    fn add_assign(&self, lhs: &mut Buffer<T, OpenCL>, rhs: &Buffer<T, OpenCL>) {
        cl_tew_self(self, lhs, rhs, "+").unwrap()
    }

    #[inline]
    fn sub_assign(&self, lhs: &mut Buffer<T, OpenCL>, rhs: &Buffer<T, OpenCL>) {
        cl_tew_self(self, lhs, rhs, "-").unwrap()
    }

    #[inline]
    fn mul_assign(&self, lhs: &mut Buffer<T, Self, ()>, rhs: &Buffer<T, Self, ()>) {
        cl_tew_self(self, lhs, rhs, "*").unwrap()
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> AssignOps<T> for custos::CUDA {
    #[inline]
    fn add_assign(&self, lhs: &mut Buffer<T, custos::CUDA>, rhs: &Buffer<T, custos::CUDA>) {
        cu_ew_self(self, lhs, rhs, "+").unwrap();
    }

    #[inline]
    fn sub_assign(&self, lhs: &mut Buffer<T, custos::CUDA>, rhs: &Buffer<T, custos::CUDA>) {
        cu_ew_self(self, lhs, rhs, "-").unwrap();
    }

    #[inline]
    fn mul_assign(&self, lhs: &mut Buffer<T, Self, ()>, rhs: &Buffer<T, Self, ()>) {
        cu_ew_self(self, lhs, rhs, "*").unwrap();
    }
}
