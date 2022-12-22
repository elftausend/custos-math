use custos::{cache::Cache, number::Number, Buffer, Device, MainMemory, CPU};

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
/// ```
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
pub trait AssignOps<T, D: Device = Self>: Device {
    /// Add assign
    /// # Examples
    /// ```
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
    fn add_assign(&self, lhs: &mut Buffer<T, Self>, rhs: &Buffer<T, D>);
    fn sub_assign(&self, lhs: &mut Buffer<T, Self>, rhs: &Buffer<T, D>);
}

pub fn ew_op<'a, T: Copy + Default, F: Fn(T, T) -> T, D: MainMemory>(
    device: &'a CPU,
    lhs: &Matrix<T, D>,
    rhs: &Matrix<T, D>,
    f: F,
) -> Matrix<'a, T> {
    let mut out = Cache::get(device, lhs.size(), [lhs.node.idx, rhs.node.idx]);
    element_wise_op_mut(lhs, rhs, &mut out, f);
    (out, lhs.dims()).into()
}

impl<T: Number, D: MainMemory> AssignOps<T, D> for CPU {
    fn add_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T, D>) {
        assign_to_lhs(lhs, rhs, |x, y| *x += y)
    }

    fn sub_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T, D>) {
        assign_to_lhs(lhs, rhs, |x, y| *x -= y)
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> AssignOps<T> for OpenCL {
    fn add_assign(&self, lhs: &mut Buffer<T, OpenCL>, rhs: &Buffer<T, OpenCL>) {
        cl_tew_self(self, lhs, rhs, "+").unwrap()
    }

    fn sub_assign(&self, lhs: &mut Buffer<T, OpenCL>, rhs: &Buffer<T, OpenCL>) {
        cl_tew_self(self, lhs, rhs, "-").unwrap()
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> AssignOps<T> for custos::CUDA {
    fn add_assign(&self, lhs: &mut Buffer<T, custos::CUDA>, rhs: &Buffer<T, custos::CUDA>) {
        cu_ew_self(self, lhs, rhs, "+").unwrap();
    }

    fn sub_assign(&self, lhs: &mut Buffer<T, custos::CUDA>, rhs: &Buffer<T, custos::CUDA>) {
        cu_ew_self(self, lhs, rhs, "-").unwrap();
    }
}
