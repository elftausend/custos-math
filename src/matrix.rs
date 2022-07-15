use std::{ffi::c_void, ops::{Add, AddAssign, Sub, Mul, SubAssign}};

use custos::{Buffer, Device, CUdeviceptr, CDatatype, get_device, GenericBlas, VecRead, number::Number};
#[cfg(feature="opencl")]
use custos::{CLDevice, opencl::{CLCache, api::{enqueue_write_buffer, wait_for_event}}};
use crate::Additional;

/// A matrix using [Buffer] described with rows and columns
/// # Example
/// The following example creates a zeroed (or values set to default) Matrix with the given dimensions.
/// ```
/// use custos::{CPU, AsDev};
/// use custos_math::Matrix;
/// 
/// let device = CPU::new().select();
/// let m = Matrix::<i32>::new(&device, (5, 8));
/// 
/// assert_eq!(m.rows(), 5);
/// assert_eq!(m.cols(), 8);
/// assert_eq!(m.size(), 5*8);
/// assert_eq!(m.read(), vec![0; 5*8])
/// ```
#[cfg_attr(not(feature="safe"), derive(Copy))]
#[derive(Clone)]
pub struct Matrix<T> {
    data: Buffer<T>,
    dims: (usize, usize)
}

impl<T> Matrix<T> {
    /// Returns an empty matrix with the specified dimensions (rows, cols).
    /// # Example
    /// ```
    /// use custos::{CPU, AsDev};
    /// use custos_math::Matrix;
    /// 
    /// let device = CPU::new().select();
    /// let m = Matrix::<f64>::new(&device, (20, 10));
    /// 
    /// assert_eq!(m.size(), 20*10);
    /// assert_eq!(m.read(), vec![0.0; 20*10])
    /// ```
    pub fn new<D: Device<T>>(device: &D, dims: (usize, usize)) -> Matrix<T> {
        Matrix {
            data: Buffer::new(device, dims.0*dims.1),
            dims,
        }
    }

    pub fn ptr(&self) -> (*mut T, *mut c_void, CUdeviceptr) {
        self.data.ptr
    }

    /// Returns a reference to the underlying buffer.
    /// # Example
    /// ```
    /// use custos::{CPU, VecRead};
    /// use custos_math::Matrix;
    /// 
    /// let device = CPU::new();
    /// let a = Matrix::from((&device, (2, 3), [1., 2., 3., 3., 2., 1.,]));
    /// let read = device.read(a.as_buf());
    /// assert_eq!(vec![1., 2., 3., 3., 2., 1.,], read);
    /// ```
    pub fn as_buf(&self) -> &Buffer<T> {
        &self.data
    }

    pub fn to_buf(self) -> Buffer<T> {
        self.data
    }

    /// Returns a mutable reference to the underlying buffer.
    pub fn as_mut_buf(&mut self) -> &mut Buffer<T> {
        &mut self.data
    }

    pub fn dims(&self) -> (usize, usize) {
        self.dims
    }

    pub fn reshape(&mut self, dims: (usize, usize)) {
        self.dims = dims;
    }

    /// Returns the row count of the matrix.
    /// 
    /// # Example
    /// ```
    /// use custos::{CPU, AsDev};
    /// use custos_math::Matrix;
    /// 
    /// let device = CPU::new().select();
    /// let matrix = Matrix::<i32>::new(&device, (2, 5));
    /// assert_eq!(matrix.rows(), 2)
    /// ```
    pub fn rows(&self) -> usize {
        self.dims.0
    }

    /// Returns the column count of the matrix.
    /// 
    /// # Example
    /// ```
    /// use custos::{CPU, AsDev};
    /// use custos_math::Matrix;
    /// 
    /// let device = CPU::new().select();
    /// let matrix = Matrix::<i32>::new(&device, (2, 5));
    /// assert_eq!(matrix.cols(), 5)
    /// ```
    pub fn cols(&self) -> usize {
        self.dims.1
    }

    /// Returns the number of elements in the matrix: rows * cols
    /// 
    /// # Example
    /// ```
    /// use custos::{CPU, AsDev};
    /// use custos_math::Matrix;
    /// 
    /// let device = CPU::new().select();
    /// let matrix = Matrix::<u16>::new(&device, (4, 12));
    /// assert_eq!(matrix.size(), 48)
    /// ```
    pub fn size(&self) -> usize {
        self.dims.0 * self.dims.1
    }

    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut_buf().as_mut_slice()
    }

    /// Matrix multiplication. Uses current global device.
    /// # Example
    /// ```
    /// use custos::{CPU, AsDev};
    /// use custos_math::Matrix;
    /// 
    /// let device = CPU::new().select();
    ///
    /// let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.,]));
    /// let b = Matrix::from((&device, (3, 2), [6., 5., 4., 3., 2., 1.,]));
    ///
    /// let c = a.gemm(&b);
    /// println!("c: {c:?}");
    ///
    /// assert_eq!(c.read(), vec![20., 14., 56., 41.,]);
    /// ```
    pub fn gemm(&self, rhs: &Matrix<T>) -> Matrix<T> 
    where T: CDatatype + GenericBlas
    {
        let device = get_device!(Gemm<T>).unwrap();
        device.gemm(self, rhs)
    }

    /* 
    /// Sets all elements to zero.
    /// # Example
    /// ```
    /// use custos::{CPU, AsDev, Matrix};
    /// 
    /// let device = CPU::new().select();
    /// let mut matrix = Matrix::from((&device, 3, 2, [4, 3, 2, 6, 9, 2,]));
    /// assert_eq!(matrix.read(), vec![4, 3, 2, 6, 9, 2]);
    /// 
    /// matrix.clear();
    /// assert_eq!(matrix.read(), vec![0; 6]);
    /// ```
    pub fn clear(&mut self) 
    where T: CDatatype
    {
        let device = get_device!(BaseOps, T).unwrap();
        device.clear(self)
    }*/

    /// Uses VecRead and current global device to read Matrix
    /// 
    /// # Example
    /// ```
    /// use custos::{CPU, AsDev};
    /// use custos_math::Matrix;
    /// 
    /// let device = CPU::new().select();
    ///
    /// let a = Matrix::from((&device, (2, 2), [5, 7, 2, 10,]));
    /// assert_eq!(a.read(), vec![5, 7, 2, 10])
    /// ```
    pub fn read(&self) -> Vec<T> 
    where T: Default + Copy 
    {
        let device = get_device!(VecRead<T>).unwrap();
        device.read(self.as_buf())
    }
}

impl<T> Default for Matrix<T> {
    fn default() -> Self {
        Self { data: Default::default(), dims: Default::default() }
    }
}

impl<T> std::ops::Deref for Matrix<T> {
    type Target = Buffer<T>;

    fn deref(&self) -> &Self::Target {
        self.as_buf()
    }
}

impl<T> std::ops::DerefMut for Matrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_buf()
    }
}

// From conversions

impl<T> From<(Buffer<T>, (usize, usize))> for Matrix<T> {
    fn from(ptr_dims: (Buffer<T>, (usize, usize))) -> Self {
        let dims = ptr_dims.1;
        Matrix {
            data: ptr_dims.0,
            dims
        }
    }
}

// no tuple for dims
impl<T> From<(Buffer<T>, usize, usize)> for Matrix<T> {
    fn from(ptr_dims: (Buffer<T>, usize, usize)) -> Self {
        let dims = (ptr_dims.1, ptr_dims.2);
        Matrix {
            data: ptr_dims.0,
            dims
        }
    }
}

// TODO: unsafe from raw parts?
#[cfg(not(feature="safe"))]
impl<T> From<(*mut T, (usize, usize))> for Matrix<T> {
    fn from(ptr_dims: (*mut T, (usize, usize))) -> Self {
        let dims = ptr_dims.1;
        Matrix {
            data: Buffer {
                ptr: (ptr_dims.0, std::ptr::null_mut(), 0), 
                len: dims.0*dims.1, 
                },
            dims
        }
    }
}

// TODO: unsafe from raw parts?
#[cfg(not(feature="safe"))]
impl<T> From<(*mut T, usize, usize)> for Matrix<T> {
    fn from(ptr_dims: (*mut T, usize, usize)) -> Self {
        Matrix {
            data: Buffer {
                ptr: (ptr_dims.0, std::ptr::null_mut(), 0), 
                len: ptr_dims.1*ptr_dims.2, 
                },
            dims: (ptr_dims.1,ptr_dims.2)
        }
    }
}

//no Weak ptr:
impl<T: Copy+Default, const N: usize> From<((usize, usize), &[T; N])> for Matrix<T> {
    fn from(dims_slice: ((usize, usize), &[T; N])) -> Self {
        let device = get_device!(Device<T>).unwrap();
        
        let buffer = Buffer::from((&device, dims_slice.1));
        Matrix {
            data: buffer,
            dims: dims_slice.0
        }        
    }
}

impl<T: Copy+Default> From<(usize, usize)> for Matrix<T> {
    fn from(dims: (usize, usize)) -> Self {
        let device = get_device!(Device<T>).unwrap();
        let buffer = Buffer::<T>::from((&device, dims.0*dims.1));
        
        Matrix {
            data: buffer,
            dims
        }
    }
}

impl<T: Copy+Default> From<(usize, usize, Vec<T>)> for Matrix<T> {
    fn from(dims_data: (usize, usize, Vec<T>)) -> Self {
        let device = get_device!(Device<T>).unwrap();
        let buffer = Buffer::<T>::from((device, dims_data.2));
        
        Matrix {
            data: buffer,
            dims: (dims_data.0, dims_data.1)
        }
    }
}

#[cfg(feature="opencl")]
impl<T> From<(&CLDevice, Matrix<T>)> for Matrix<T> {
    fn from(device_matrix: (&CLDevice, Matrix<T>)) -> Self {
        //assert!(CPU_CACHE.with(|cache| !cache.borrow().nodes.is_empty()), "no allocations");
        let y = CLCache::get::<T>(device_matrix.0, device_matrix.1.size());
        let event = unsafe {enqueue_write_buffer(&device_matrix.0.queue(), y.ptr.1, &device_matrix.1, true).unwrap()};
        wait_for_event(event).unwrap();
        Matrix::from((y, device_matrix.1.dims()))
    }
}

use crate::{Gemm, BaseOps, AssignOps};
#[cfg(feature="cuda")]
use custos::{CudaDevice, cuda::{CudaCache, api::cu_write}};

#[cfg(feature="cuda")]
impl<T> From<(&CudaDevice, Matrix<T>)> for Matrix<T> {
    fn from(device_matrix: (&CudaDevice, Matrix<T>)) -> Self {
        let dst = CudaCache::get::<T>(device_matrix.0, device_matrix.1.size());
        cu_write(dst.ptr.2, &device_matrix.1).unwrap();
        Matrix::from((dst, device_matrix.1.dims()))
    }
}

impl<T: Copy, D: Device<T>, const N: usize> From<(&D, (usize, usize), [T; N])> for Matrix<T> {
    fn from(dims_slice: (&D, (usize, usize), [T; N])) -> Self {
        let buffer = Buffer::from((dims_slice.0, dims_slice.2));
        Matrix {
            data: buffer,
            dims: dims_slice.1
        }        
    }
}

// no tuple for dims
impl<T: Copy, D: Device<T>, const N: usize> From<(&D, usize, usize, [T; N])> for Matrix<T> {
    fn from(dims_slice: (&D, usize, usize, [T; N])) -> Self {
        let buffer = Buffer::from((dims_slice.0, dims_slice.3));
        Matrix {
            data: buffer,
            dims: (dims_slice.1, dims_slice.2)
        }        
    }
}

impl<T: Copy, D: Device<T>> From<(&D, (usize, usize), Vec<T>)> for Matrix<T> {
    fn from(dims_slice: (&D, (usize, usize), Vec<T>)) -> Self {
        let buffer = Buffer::from((dims_slice.0, dims_slice.2));
        Matrix {
            data: buffer,
            dims: dims_slice.1
        }        
    }
}

// no tuple for dims
impl<T: Copy, D: Device<T>> From<(&D, usize, usize, Vec<T>)> for Matrix<T> {
    fn from(dims_slice: (&D, usize, usize, Vec<T>)) -> Self {
        let buffer = Buffer::from((dims_slice.0, dims_slice.3));
        Matrix {
            data: buffer,
            dims: (dims_slice.1, dims_slice.2)
        }        
    }
}


impl<T: Copy, D: Device<T>> From<(&D, (usize, usize), &[T])> for Matrix<T> {
    fn from(dims_slice: (&D, (usize, usize), &[T])) -> Self {
        let buffer = Buffer::from((dims_slice.0, dims_slice.2));
        Matrix {
            data: buffer,
            dims: dims_slice.1
        }        
    }
}

// no tuple for dims
impl<T: Copy, D: Device<T>> From<(&D, usize, usize, &[T])> for Matrix<T> {
    fn from(dims_slice: (&D, usize, usize, &[T])) -> Self {
        let buffer = Buffer::from((dims_slice.0, dims_slice.3));
        Matrix {
            data: buffer,
            dims: (dims_slice.1, dims_slice.2)
        }        
    }
}

impl<T: Copy, D: Device<T>> From<(&D, (usize, usize), &Vec<T>)> for Matrix<T> {
    fn from(dims_slice: (&D, (usize, usize), &Vec<T>)) -> Self {
        let buffer = Buffer::from((dims_slice.0, dims_slice.2));
        Matrix {
            data: buffer,
            dims: dims_slice.1
        }
    }
}

//-------------Add-------------

impl<T: CDatatype> Add<Self> for &Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.add(self, rhs)
    }
}

impl<T: CDatatype> Add<Self> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.add(&self, &rhs)
    }
}

impl<T: CDatatype> Add<&Self> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: &Self) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.add(&self, rhs)
    }
}

impl<T: CDatatype> Add<Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.add(self, &rhs)
    }
}

impl<T: CDatatype> Add<T> for &Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: T) -> Self::Output {
        self.adds(rhs)
    }
}

impl<T: CDatatype> Add<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: T) -> Self::Output {
        self.adds(rhs)
    }
}

//-------------Sub-------------

impl<T: CDatatype> Sub<Self> for &Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.sub(self, rhs)
    }
}

impl<T: CDatatype> Sub<Self> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.sub(&self, &rhs)
    }
}

impl<T: CDatatype> Sub<&Self> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: &Self) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.sub(&self, rhs)
    }
}

impl<T: CDatatype> Sub<Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.sub(self, &rhs)
    }
}

impl<T: CDatatype> Sub<T> for &Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, _rhs: T) -> Self::Output {
        todo!()
        //self.subs(rhs)
    }
}

impl<T: CDatatype> Sub<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, _rhs: T) -> Self::Output {
        todo!()
        //self.subs(rhs)
    }
}

//-------------Mul-------------

impl<T: CDatatype> Mul<Self> for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.mul(self, rhs)
    }
}

impl<T: CDatatype> Mul<Self> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.mul(&self, &rhs)
    }
}

impl<T: CDatatype> Mul<&Self> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: &Self) -> Self::Output {
        let device = get_device!(BaseOps<T>).unwrap();
        device.mul(&self, rhs)
    }
}

impl<T: CDatatype> AddAssign<&Matrix<T>> for Matrix<T> {
    fn add_assign(&mut self, rhs: &Matrix<T>) {
        let device = get_device!(AssignOps<T>).unwrap();
        device.add_assign(self, rhs)
    }
}

impl<T: CDatatype> SubAssign<&Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        let device = get_device!(AssignOps<T>).unwrap();
        device.sub_assign(self, rhs)
    }
}

impl<T: Number> core::fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let data = self.read();

        writeln!(f, "dims={:?}", self.dims)?;
        write!(f, "[")?;
        
        let max = self.dims.0*self.dims.1;
        for (count, value) in data.iter().enumerate() {
            write!(f, "{:?}, ", value)?;
        
            if (count+1) % self.dims.1 == 0 && count+1 != max {
                writeln!(f)?;
            }            
        }       
        write!(f, ":datatype={}]", core::any::type_name::<T>())
    }
}
