use custos::{Buffer, CPU, cpu::CPUCache, number::Number};

#[cfg(any(feature="cuda", feature="opencl"))]
use custos::CDatatype;

#[cfg(feature="opencl")]
use custos::CLDevice;
#[cfg(feature="opencl")]
use crate::opencl::cl_tew_self;

use crate::{Matrix, assign_to_lhs, element_wise_op_mut};
#[cfg(feature="cuda")]
use crate::cu_ew_self;

/// Assignment operations
/// # Examples
/// ```
/// use custos::{CPU, VecRead};
/// use custos_math::{Matrix, AssignOps};
/// 
/// let device = CPU::new();
/// let mut lhs = Matrix::from((&device, 2, 2, [3, 5, 4, 1]));
/// let rhs = Matrix::from((&device, 2, 2, [1, 8, 6, 2]));
/// 
/// device.add_assign(&mut lhs, &rhs);
/// assert_eq!(vec![4, 13, 10, 3], device.read(lhs.as_buf()));
/// 
/// device.sub_assign(&mut lhs, &rhs);
/// assert_eq!(vec![3, 5, 4, 1], device.read(lhs.as_buf()));
/// ```
pub trait AssignOps<T> {
    /// Add assign
    /// # Examples
    /// ```
    /// use custos::{CPU, VecRead};
    /// use custos_math::{Matrix, AssignOps};
    /// 
    /// let device = CPU::new();
    /// let mut lhs = Matrix::from((&device, 2, 2, [3, 5, 4, 1]));
    /// let rhs = Matrix::from((&device, 2, 2, [1, 8, 6, 2]));
    /// 
    /// device.add_assign(&mut lhs, &rhs);
    /// assert_eq!(vec![4, 13, 10, 3], device.read(lhs.as_buf()));
    /// 
    /// device.sub_assign(&mut lhs, &rhs);
    /// assert_eq!(vec![3, 5, 4, 1], device.read(lhs.as_buf()));
    /// ```
    fn add_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>);
    fn sub_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>);
}

pub fn ew_op<T: Copy+Default, F: Fn(T, T) -> T>(device: &CPU, lhs: &Matrix<T>, rhs: &Matrix<T>, f: F) -> Matrix<T> {
    let mut out = CPUCache::get::<T>(device, lhs.size());
    element_wise_op_mut(lhs, rhs, &mut out, f);
    (out, lhs.dims()).into()
}

impl<T: Number> AssignOps<T> for CPU {
    fn add_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>) {
        assign_to_lhs(lhs, rhs, |x, y| *x += y)
    }

    fn sub_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>) {
        assign_to_lhs(lhs, rhs, |x, y| *x -= y)
    }
}

#[cfg(feature="opencl")]
impl<T: CDatatype> AssignOps<T> for CLDevice {
    fn add_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>) {
        cl_tew_self(self, lhs, rhs, "+").unwrap()
    }

    fn sub_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>) {
        cl_tew_self(self, lhs, rhs, "-").unwrap()
    }
}

#[cfg(feature="cuda")]
impl<T: CDatatype> AssignOps<T> for custos::CudaDevice {
    fn add_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>) {
        cu_ew_self(self, lhs, rhs, "+").unwrap();
    }

    fn sub_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>) {
        cu_ew_self(self, lhs, rhs, "-").unwrap();
    }
}