use std::{
    ffi::c_void,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Mul, Sub, SubAssign},
};

use crate::{AssignOps, BaseOps, Gemm};

use custos::{
    get_device, Alloc, BufFlag, Buffer, CDatatype, CUdeviceptr, Device, GenericBlas, GraphReturn,
    Node, VecRead,
};
#[cfg(feature = "opencl")]
use custos::{
    opencl::api::{enqueue_write_buffer, wait_for_event},
    CLDevice,
};

#[cfg(feature = "cuda")]
use custos::{cuda::api::cu_write, CudaDevice};

#[cfg(any(feature = "cuda", feature = "opencl"))]
use custos::cache::Cache;

/// A matrix using [Buffer] described with rows and columns
/// # Example
/// The following example creates a zeroed (or values set to default) Matrix with the given dimensions.
/// ```
/// use custos::CPU;
/// use custos_math::Matrix;
///
/// let device = CPU::new();
/// let m = Matrix::<i32>::new(&device, (5, 8));
///
/// assert_eq!(m.rows(), 5);
/// assert_eq!(m.cols(), 8);
/// assert_eq!(m.size(), 5*8);
/// assert_eq!(m.read(), vec![0; 5*8])
/// ```
pub struct Matrix<'a, T> {
    pub data: Buffer<'a, T>,
    pub dims: (usize, usize),
}

impl<'a, T> Matrix<'a, T> {
    /// Returns an empty matrix with the specified dimensions (rows, cols).
    /// # Example
    /// ```
    /// use custos::CPU;
    /// use custos_math::Matrix;
    ///
    /// let device = CPU::new();
    /// let m = Matrix::<f64>::new(&device, (20, 10));
    ///
    /// assert_eq!(m.size(), 20*10);
    /// assert_eq!(m.read(), vec![0.0; 20*10])
    /// ```
    pub fn new<D: Alloc<T> + GraphReturn>(device: &D, dims: (usize, usize)) -> Matrix<T> {
        Matrix {
            data: Buffer::new(device, dims.0 * dims.1),
            dims,
        }
    }

    #[inline]
    pub fn device(&self) -> Device {
        self.data.device
    }

    #[inline]
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
    #[inline]
    pub fn as_buf(&self) -> &Buffer<'a, T> {
        &self.data
    }

    #[inline]
    pub fn to_buf(self) -> Buffer<'a, T> {
        self.data
    }

    /// Returns a mutable reference to the underlying buffer.
    #[inline]
    pub fn as_mut_buf(&mut self) -> &mut Buffer<'a, T> {
        &mut self.data
    }

    #[inline]
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
    /// use custos::CPU;
    /// use custos_math::Matrix;
    ///
    /// let device = CPU::new();
    /// let matrix = Matrix::<i32>::new(&device, (2, 5));
    /// assert_eq!(matrix.rows(), 2)
    /// ```
    #[inline]
    pub fn rows(&self) -> usize {
        self.dims.0
    }

    /// Returns the column count of the matrix.
    ///
    /// # Example
    /// ```
    /// use custos::CPU;
    /// use custos_math::Matrix;
    ///
    /// let device = CPU::new();
    /// let matrix = Matrix::<i32>::new(&device, (2, 5));
    /// assert_eq!(matrix.cols(), 5)
    /// ```
    #[inline]
    pub fn cols(&self) -> usize {
        self.dims.1
    }

    /// Returns the number of elements in the matrix: rows * cols
    ///
    /// # Example
    /// ```
    /// use custos::CPU;
    /// use custos_math::Matrix;
    ///
    /// let device = CPU::new();
    /// let matrix = Matrix::<u16>::new(&device, (4, 12));
    /// assert_eq!(matrix.size(), 48)
    /// ```
    #[inline]
    pub fn size(&self) -> usize {
        self.dims.0 * self.dims.1
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut_buf().as_mut_slice()
    }

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
    pub fn gemm<'b>(&self, rhs: &Matrix<'b, T>) -> Matrix<'a, T>
    where
        T: CDatatype + GenericBlas,
    {
        let device = get_device!(self.device(), Gemm<T>);
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
    /// use custos::CPU;
    /// use custos_math::Matrix;
    ///
    /// let device = CPU::new();
    ///
    /// let a = Matrix::from((&device, (2, 2), [5, 7, 2, 10,]));
    /// assert_eq!(a.read(), vec![5, 7, 2, 10])
    /// ```
    pub fn read(&self) -> Vec<T>
    where
        T: Default + Copy,
    {
        let device = get_device!(self.device(), VecRead<T>);
        device.read(self.as_buf())
    }

    /// Creates a shallow copy of &self.
    pub fn shallow(&self) -> Matrix<'a, T> {
        unsafe {
            Self {
                data: self.data.shallow(),
                dims: self.dims,
            }
        }
    }

    /// Creates a shallow copy or a deep copy of &self, depening on whether the `realloc` feature is activated.
    pub fn shallow_or_clone(&self) -> Matrix<'a, T>
    where
        T: Clone,
    {
        unsafe {
            Self {
                data: self.data.shallow_or_clone(),
                dims: self.dims,
            }
        }
    }
}

impl<T> Default for Matrix<'_, T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
            dims: Default::default(),
        }
    }
}

impl<'a, T> std::ops::Deref for Matrix<'a, T> {
    type Target = Buffer<'a, T>;

    fn deref(&self) -> &Self::Target {
        self.as_buf()
    }
}

impl<'a, T> std::ops::DerefMut for Matrix<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_buf()
    }
}

impl<'a, T: Clone> Clone for Matrix<'a, T> {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), dims: self.dims.clone() }
    }
}

// From conversions

impl<'a, T> From<(Buffer<'a, T>, (usize, usize))> for Matrix<'a, T> {
    #[inline]
    fn from(ptr_dims: (Buffer<'a, T>, (usize, usize))) -> Self {
        let dims = ptr_dims.1;
        Matrix {
            data: ptr_dims.0,
            dims,
        }
    }
}

// no tuple for dims
impl<'a, T> From<(Buffer<'a, T>, usize, usize)> for Matrix<'a, T> {
    #[inline]
    fn from(ptr_dims: (Buffer<'a, T>, usize, usize)) -> Self {
        let dims = (ptr_dims.1, ptr_dims.2);
        Matrix {
            data: ptr_dims.0,
            dims,
        }
    }
}

// TODO: unsafe from raw parts?
// is wrapper flag ok?
#[cfg(not(feature = "safe"))]
impl<T> From<(*mut T, (usize, usize))> for Matrix<'_, T> {
    #[inline]
    fn from(ptr_dims: (*mut T, (usize, usize))) -> Self {
        let dims = ptr_dims.1;
        Matrix {
            data: Buffer {
                ptr: (ptr_dims.0, std::ptr::null_mut(), 0),
                len: dims.0 * dims.1,
                // Mind default device, this will not work
                device: Default::default(),
                flag: BufFlag::Wrapper,
                node: Node::default(),
                p: PhantomData,
            },
            dims,
        }
    }
}

// TODO: unsafe from raw parts?
// is wrapper flag ok?
#[cfg(not(feature = "safe"))]
impl<T> From<(*mut T, usize, usize)> for Matrix<'_, T> {
    #[inline]
    fn from(ptr_dims: (*mut T, usize, usize)) -> Self {
        Matrix {
            data: Buffer {
                ptr: (ptr_dims.0, std::ptr::null_mut(), 0),
                len: ptr_dims.1 * ptr_dims.2,
                device: Device::default(),
                flag: BufFlag::Wrapper,
                node: Node::default(),
                p: PhantomData,
            },
            dims: (ptr_dims.1, ptr_dims.2),
        }
    }
}

/*
impl<T: Copy + Default, const N: usize> From<((usize, usize), &[T; N])> for Matrix<'_, T> {
    fn from(dims_slice: ((usize, usize), &[T; N])) -> Self {
        let device = get_device!(Device<T>).unwrap();

        let buffer = Buffer::from((&device, dims_slice.1));
        Matrix {
            data: buffer,
            dims: dims_slice.0,
        }
    }
}*/

/*
impl<T: Copy + Default> From<(usize, usize)> for Matrix<'_, T> {
    fn from(dims: (usize, usize)) -> Self {
        let device = get_device!(Device<T>).unwrap();
        let buffer = Buffer::<T>::from((&device, dims.0 * dims.1));

        Matrix { data: buffer, dims }
    }
}

impl<T: Copy + Default> From<(usize, usize, Vec<T>)> for Matrix<'_, T> {
    fn from(dims_data: (usize, usize, Vec<T>)) -> Self {
        let device = get_device!(Device<T>).unwrap();
        let buffer = Buffer::<T>::from((device, dims_data.2));

        Matrix {
            data: buffer,
            dims: (dims_data.0, dims_data.1),
        }
    }
}*/

#[cfg(feature = "opencl")]
impl<'a, 'b, T> From<(&'a CLDevice, Matrix<'b, T>)> for Matrix<'a, T> {
    fn from(device_matrix: (&'a CLDevice, Matrix<'b, T>)) -> Self {
        //assert!(CPU_CACHE.with(|cache| !cache.borrow().nodes.is_empty()), "no allocations");
        let out = Cache::get::<T, _>(device_matrix.0, device_matrix.1.size(), ());
        let event = unsafe {
            enqueue_write_buffer(&device_matrix.0.queue(), out.ptr.1, &device_matrix.1, true)
                .unwrap()
        };
        wait_for_event(event).unwrap();
        Matrix::from((out, device_matrix.1.dims()))
    }
}

#[cfg(feature = "cuda")]
impl<'a, 'b, T> From<(&'a CudaDevice, Matrix<'b, T>)> for Matrix<'a, T> {
    fn from(device_matrix: (&'a CudaDevice, Matrix<'b, T>)) -> Self {
        let dst = Cache::get(device_matrix.0, device_matrix.1.size(), ());
        cu_write(dst.ptr.2, &device_matrix.1).unwrap();
        Matrix::from((dst, device_matrix.1.dims()))
    }
}

impl<'a, T: Copy, D: Alloc<T> + GraphReturn + ?Sized, const N: usize>
    From<(&'a D, (usize, usize), [T; N])> for Matrix<'a, T>
{
    fn from(dims_slice: (&'a D, (usize, usize), [T; N])) -> Self {
        let data = Buffer::from((dims_slice.0, dims_slice.2));
        Matrix {
            data,
            dims: dims_slice.1,
        }
    }
}

impl<'a, T: Copy, D: Alloc<T> + GraphReturn + ?Sized> From<(&'a D, usize, usize)>
    for Matrix<'a, T>
{
    fn from(device_dims: (&'a D, usize, usize)) -> Self {
        let data = Buffer::new(device_dims.0, device_dims.1 * device_dims.2);
        Matrix {
            data,
            dims: (device_dims.1, device_dims.2),
        }
    }
}

impl<'a, T: Copy, D: Alloc<T> + GraphReturn + ?Sized> From<(&'a D, (usize, usize))>
    for Matrix<'a, T>
{
    fn from(device_dims: (&'a D, (usize, usize))) -> Self {
        let data = Buffer::new(device_dims.0, device_dims.1 .0 * device_dims.1 .1);
        Matrix {
            data,
            dims: device_dims.1,
        }
    }
}

// no tuple for dims
impl<'a, T: Copy, D: Alloc<T> + GraphReturn + ?Sized, const N: usize>
    From<(&'a D, usize, usize, [T; N])> for Matrix<'a, T>
{
    fn from(dims_slice: (&'a D, usize, usize, [T; N])) -> Self {
        let data = Buffer::from((dims_slice.0, dims_slice.3));
        Matrix {
            data,
            dims: (dims_slice.1, dims_slice.2),
        }
    }
}

impl<'a, T: Copy, D: Alloc<T> + GraphReturn + ?Sized> From<(&'a D, (usize, usize), Vec<T>)>
    for Matrix<'a, T>
{
    fn from(dims_slice: (&'a D, (usize, usize), Vec<T>)) -> Self {
        let data = Buffer::from((dims_slice.0, dims_slice.2));
        Matrix {
            data,
            dims: dims_slice.1,
        }
    }
}

// no tuple for dims
impl<'a, T: Copy, D: Alloc<T> + GraphReturn + ?Sized> From<(&'a D, usize, usize, Vec<T>)>
    for Matrix<'a, T>
{
    fn from(dims_slice: (&'a D, usize, usize, Vec<T>)) -> Self {
        let data = Buffer::from((dims_slice.0, dims_slice.3));
        Matrix {
            data,
            dims: (dims_slice.1, dims_slice.2),
        }
    }
}

impl<'a, T: Copy, D: Alloc<T> + GraphReturn + ?Sized> From<(&'a D, (usize, usize), &[T])>
    for Matrix<'a, T>
{
    fn from(dims_slice: (&'a D, (usize, usize), &[T])) -> Self {
        let data = Buffer::from((dims_slice.0, dims_slice.2));
        Matrix {
            data,
            dims: dims_slice.1,
        }
    }
}

// no tuple for dims
impl<'a, T: Copy, D: Alloc<T> + GraphReturn + ?Sized> From<(&'a D, usize, usize, &[T])>
    for Matrix<'a, T>
{
    fn from(dims_slice: (&'a D, usize, usize, &[T])) -> Self {
        let data = Buffer::from((dims_slice.0, dims_slice.3));
        Matrix {
            data,
            dims: (dims_slice.1, dims_slice.2),
        }
    }
}

impl<'a, T: Copy, D: Alloc<T> + GraphReturn + ?Sized> From<(&'a D, (usize, usize), &Vec<T>)>
    for Matrix<'a, T>
{
    fn from(dims_slice: (&'a D, (usize, usize), &Vec<T>)) -> Self {
        let data = Buffer::from((dims_slice.0, dims_slice.2));
        Matrix {
            data,
            dims: dims_slice.1,
        }
    }
}

//-------------Add-------------

impl<'a, T: CDatatype> Add<Self> for &Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn add(self, rhs: Self) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.add(self, rhs)
    }
}

impl<'a, T: CDatatype> Add<Self> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn add(self, rhs: Self) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.add(&self, &rhs)
    }
}

impl<'a, T: CDatatype> Add<&Self> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn add(self, rhs: &Self) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.add(&self, rhs)
    }
}

impl<'a, T: CDatatype> Add<Matrix<'a, T>> for &Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.add(self, &rhs)
    }
}

impl<'a, T: CDatatype> Add<T> for &Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn add(self, rhs: T) -> Self::Output {
        self.adds(rhs)
    }
}

impl<'a, T: CDatatype> Add<T> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn add(self, rhs: T) -> Self::Output {
        self.adds(rhs)
    }
}

//-------------Sub-------------

impl<'a, T: CDatatype> Sub<Self> for &Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.sub(self, rhs)
    }
}

impl<'a, T: CDatatype> Sub<Self> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.sub(&self, &rhs)
    }
}

impl<'a, T: CDatatype> Sub<&Self> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn sub(self, rhs: &Self) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.sub(&self, rhs)
    }
}

impl<'a, T: CDatatype> Sub<Matrix<'a, T>> for &Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.sub(self, &rhs)
    }
}

impl<'a, T: CDatatype> Sub<T> for &Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn sub(self, _rhs: T) -> Self::Output {
        todo!()
        //self.subs
    }
}

impl<'a, T: CDatatype> Sub<T> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn sub(self, _rhs: T) -> Self::Output {
        todo!()
        //self.subs(rhs)
    }
}

//-------------Mul-------------

impl<'a, T: CDatatype> Mul<Self> for &Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.mul(self, rhs)
    }
}

impl<'a, T: CDatatype> Mul<Self> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.mul(&self, &rhs)
    }
}

impl<'a, T: CDatatype> Mul<&Self> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn mul(self, rhs: &Self) -> Self::Output {
        let device = get_device!(self.device(), BaseOps<T>);
        device.mul(&self, rhs)
    }
}

impl<'a, T: CDatatype> Mul<T> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.muls(rhs)
    }
}

impl<'a, T: CDatatype> Mul<&T> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn mul(self, rhs: &T) -> Self::Output {
        self.muls(*rhs)
    }
}

impl<'a, T: CDatatype> Mul<T> for &Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.muls(rhs)
    }
}

// div

impl<'a, T: CDatatype> Div<T> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn div(self, rhs: T) -> Self::Output {
        self.divs(rhs)
    }
}

impl<'a, T: CDatatype> Div<T> for &Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn div(self, rhs: T) -> Self::Output {
        self.divs(rhs)
    }
}

impl<T: CDatatype> AddAssign<&Matrix<'_, T>> for Matrix<'_, T> {
    fn add_assign(&mut self, rhs: &Matrix<T>) {
        let device = get_device!(self.device(), AssignOps<T>);
        device.add_assign(self, rhs)
    }
}

impl<T: CDatatype> AddAssign<Matrix<'_, T>> for Matrix<'_, T> {
    fn add_assign(&mut self, rhs: Matrix<T>) {
        let device = get_device!(self.device(), AssignOps<T>);
        device.add_assign(self, &rhs)
    }
}

impl<T: CDatatype> SubAssign<&Matrix<'_, T>> for &mut Matrix<'_, T> {
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        let device = get_device!(self.device(), AssignOps<T>);
        device.sub_assign(self, rhs)
    }
}

impl<T: CDatatype> SubAssign<Matrix<'_, T>> for &mut Matrix<'_, T> {
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        let device = get_device!(self.device(), AssignOps<T>);
        device.sub_assign(self, &rhs)
    }
}

impl<T: CDatatype> SubAssign<&Matrix<'_, T>> for Matrix<'_, T> {
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        let device = get_device!(self.device(), AssignOps<T>);
        device.sub_assign(self, rhs)
    }
}

impl<T: CDatatype> SubAssign<Matrix<'_, T>> for Matrix<'_, T> {
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        let device = get_device!(self.device(), AssignOps<T>);
        device.sub_assign(self, &rhs)
    }
}

impl<T: Default + Copy + core::fmt::Debug> core::fmt::Debug for Matrix<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let data = self.read();

        writeln!(f, "dims={:?}", self.dims)?;
        write!(f, "[")?;

        let max = self.dims.0 * self.dims.1;
        for (count, value) in data.iter().enumerate() {
            write!(f, "{:?}, ", value)?;

            if (count + 1) % self.dims.1 == 0 && count + 1 != max {
                writeln!(f)?;
            }
        }
        write!(f, ":datatype={}]", core::any::type_name::<T>())
    }
}
