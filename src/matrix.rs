use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

use crate::{AdditionalOps, AssignOps, BaseOps};

#[cfg(feature = "opencl")]
use custos::{
    opencl::api::{enqueue_write_buffer, wait_for_event},
    OpenCL,
};
use custos::{
    Alloc, Buffer, CloneBuf, Device, IsShapeIndep, MainMemory, Read, ShallowCopy, Shape, ToDim, CPU,
};

#[cfg(feature = "cuda")]
use custos::{cuda::api::cu_write, CUDA};

mod impl_with_shape;

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
pub struct Matrix<'a, T = f32, D: Device = CPU, S: Shape = ()> {
    pub data: Buffer<'a, T, D, S>,
    pub dims: (usize, usize),
}

impl<'a, T, D: Device, S: Shape> Matrix<'a, T, D, S> {
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
    #[inline]
    pub fn new(device: &'a D, dims: (usize, usize)) -> Matrix<'a, T, D, S>
    where
        D: Alloc<'a, T, S>,
    {
        Matrix {
            data: Buffer::new(device, dims.0 * dims.1),
            dims,
        }
    }

    #[inline]
    pub fn device(&self) -> &'a D {
        self.data.device()
    }

    // TODO: mind ptrs_Mut
    //#[inline]
    //pub fn ptr(&self) -> (*mut T, *mut c_void, u64)
    //where D::Ptr<T, ()>: CommonPtrs<T>
    //{
    //    self.data.ptrs_mut()
    //}

    /// Returns a reference to the underlying buffer.
    /// # Example
    /// ```
    /// use custos::{CPU, Read};
    /// use custos_math::Matrix;
    ///
    /// let device = CPU::new();
    /// let a = Matrix::from((&device, (2, 3), [1., 2., 3., 3., 2., 1.,]));
    /// let read = a.read();
    /// assert_eq!(vec![1., 2., 3., 3., 2., 1.,], read);
    /// ```
    #[inline]
    pub fn as_buf(&self) -> &Buffer<'a, T, D, S> {
        &self.data
    }

    #[inline]
    pub fn to_buf(self) -> Buffer<'a, T, D, S> {
        self.data
    }

    /// Returns a mutable reference to the underlying buffer.
    #[inline]
    pub fn as_buf_mut(&mut self) -> &mut Buffer<'a, T, D, S> {
        &mut self.data
    }

    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        self.dims
    }

    #[inline]
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
    pub fn as_slice(&self) -> &[T]
    where
        D: MainMemory,
    {
        self.data.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T]
    where
        D: MainMemory,
    {
        self.as_buf_mut().as_mut_slice()
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
    pub fn read(&'a self) -> D::Read<'a>
    where
        T: Default + Copy,
        D: Read<T, D, S>,
    {
        self.device().read(self.as_buf())
    }

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
    #[cfg(not(feature = "no-std"))]
    pub fn read_to_vec(&self) -> Vec<T>
    where
        T: Default + Copy,
        D: Read<T, D, S>,
    {
        self.device().read_to_vec(self.as_buf())
    }

    /// Creates a shallow copy of &self.
    pub fn shallow(&self) -> Matrix<'a, T, D, S>
    where
        D::Ptr<T, S>: ShallowCopy,
    {
        unsafe {
            Self {
                data: self.data.shallow(),
                dims: self.dims,
            }
        }
    }

    /// Creates a shallow copy or a deep copy of &self, depening on whether the `realloc` feature is activated.
    pub fn shallow_or_clone(&self) -> Matrix<'a, T, D, S>
    where
        T: Clone,
        D::Ptr<T, S>: ShallowCopy,
        D: CloneBuf<'a, T, S>,
    {
        unsafe {
            Self {
                data: self.data.shallow_or_clone(),
                dims: self.dims,
            }
        }
    }
}

impl<T, D: Device> Default for Matrix<'_, T, D>
where
    D::Ptr<T, ()>: Default,
{
    fn default() -> Self {
        Self {
            data: Default::default(),
            dims: Default::default(),
        }
    }
}

impl<'a, T, D: Device, S: Shape> Matrix<'a, T, D, S> {
    /// Converts a (non stack allocated) `Buffer` with no shape to a `Buffer` with shape `O`.
    #[inline]
    pub fn to_dims<O: Shape>(self) -> Matrix<'a, T, D, O>
    where
        D: ToDim<T, S, O>,
    {
        let data = self.data.to_dims();

        Matrix {
            data,
            dims: self.dims,
        }
    }
}

impl<T, D: IsShapeIndep, S: Shape> Matrix<'_, T, D, S> {
    #[inline]
    pub fn as_dims<'b, O: Shape>(&self) -> &Matrix<'b, T, D, O> {
        unsafe { &*(self as *const Self).cast() }
    }

    #[inline]
    pub fn as_dims_mut<'b, O: Shape>(&mut self) -> &mut Matrix<'b, T, D, O> {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<'a, T, D: Device, S: Shape> core::ops::Deref for Matrix<'a, T, D, S> {
    type Target = Buffer<'a, T, D, S>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_buf()
    }
}

impl<'a, T, D: Device, S: Shape> core::ops::DerefMut for Matrix<'a, T, D, S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_buf_mut()
    }
}

impl<'a, T, S, D> Clone for Matrix<'a, T, D, S>
where
    T: Clone,
    S: Shape,
    D: CloneBuf<'a, T, S>,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            dims: self.dims.clone(),
        }
    }
}

// From conversions

impl<'a, T, D: Device, S: Shape> From<(Buffer<'a, T, D, S>, (usize, usize))>
    for Matrix<'a, T, D, S>
{
    #[inline]
    fn from((data, dims): (Buffer<'a, T, D, S>, (usize, usize))) -> Self {
        Matrix { data, dims }
    }
}

// no tuple for dims
impl<'a, T, D: Device, S: Shape> From<(Buffer<'a, T, D, S>, usize, usize)> for Matrix<'a, T, D, S> {
    #[inline]
    fn from((data, rows, cols): (Buffer<'a, T, D, S>, usize, usize)) -> Self {
        Matrix {
            data,
            dims: (rows, cols),
        }
    }
}

// TODO: unsafe from raw parts?
#[cfg(feature = "cpu")]
impl<'a, T> From<(*mut T, (usize, usize))> for Matrix<'a, T> {
    #[inline]
    fn from((ptr, dims): (*mut T, (usize, usize))) -> Self {
        unsafe {
            Matrix {
                data: Buffer::from_raw_host(ptr, dims.0 * dims.1),
                dims,
            }
        }
    }
}

// TODO: unsafe from raw parts?
#[cfg(feature = "cpu")]
impl<'a, T> From<(&'a CPU, *mut T, (usize, usize))> for Matrix<'a, T> {
    #[inline]
    fn from((cpu, ptr, dims): (&'a CPU, *mut T, (usize, usize))) -> Self {
        unsafe {
            Matrix {
                data: Buffer::from_raw_host_device(cpu, ptr, dims.0 * dims.1),
                dims,
            }
        }
    }
}

/*
// TODO: unsafe from raw parts?
// is wrapper flag ok?
#[cfg(not(feature = "safe"))]
impl<T, D: Device> From<(*mut T, usize, usize)> for Matrix<'_, T, D> {
    #[inline]
    fn from(ptr_dims: (*mut T, usize, usize)) -> Self {
        Matrix {
            data: Buffer {
                ptr: (ptr_dims.0, std::ptr::null_mut(), 0),
                len: ptr_dims.1 * ptr_dims.2,
                device: Device::default(),
                flag: BufFlag::Wrapper,
                node: Node::default(),
            },
            dims: (ptr_dims.1, ptr_dims.2),
        }
    }
}*/

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
impl<'a, 'b, T> From<(&'a OpenCL, Matrix<'b, T>)> for Matrix<'a, T, OpenCL> {
    fn from((device, matrix): (&'a OpenCL, Matrix<'b, T>)) -> Self {
        //assert!(CPU_CACHE.with(|cache| !cache.borrow().nodes.is_empty()), "no allocations");
        let out = device.retrieve(matrix.size(), ());

        let event =
            unsafe { enqueue_write_buffer(&device.queue(), out.ptr.ptr, &matrix, true).unwrap() };
        wait_for_event(event).unwrap();
        Matrix::from((out, matrix.dims()))
    }
}

#[cfg(feature = "cuda")]
impl<'a, 'b, T> From<(&'a CUDA, Matrix<'b, T>)> for Matrix<'a, T, CUDA> {
    fn from(device_matrix: (&'a CUDA, Matrix<'b, T>)) -> Self {
        let dst = device_matrix.0.retrieve(device_matrix.1.size(), ());
        cu_write(dst.ptr.ptr, &device_matrix.1).unwrap();
        Matrix::from((dst, device_matrix.1.dims()))
    }
}

impl<'a, T: Copy, D: Alloc<'a, T> + IsShapeIndep, const N: usize>
    From<(&'a D, (usize, usize), [T; N])> for Matrix<'a, T, D>
{
    fn from((device, dims, slice): (&'a D, (usize, usize), [T; N])) -> Self {
        let data = Buffer::from((device, slice));
        Matrix { data, dims }
    }
}

// no tuple for dims
impl<'a, T: Copy, D: Alloc<'a, T> + IsShapeIndep, const N: usize>
    From<(&'a D, usize, usize, [T; N])> for Matrix<'a, T, D>
{
    fn from(dims_slice: (&'a D, usize, usize, [T; N])) -> Self {
        let data = Buffer::from((dims_slice.0, dims_slice.3));
        Matrix {
            data,
            dims: (dims_slice.1, dims_slice.2),
        }
    }
}

impl<'a, T: Copy, D: Alloc<'a, T>> From<(&'a D, usize, usize)> for Matrix<'a, T, D> {
    fn from((device, rows, cols): (&'a D, usize, usize)) -> Self {
        let data = Buffer::new(device, rows * cols);
        Matrix {
            data,
            dims: (rows, cols),
        }
    }
}

impl<'a, T: Copy, D: Alloc<'a, T>> From<(&'a D, (usize, usize))> for Matrix<'a, T, D> {
    fn from((device, dims): (&'a D, (usize, usize))) -> Self {
        let data = Buffer::new(device, dims.0 * dims.1);
        Matrix { data, dims }
    }
}

// FIXME: In this case, GraphReturn acts as an "IsDynamic" trait, as GraphReturn is not implemented for Stack
#[cfg(not(feature = "no-std"))]
impl<'a, T: Copy, D: Alloc<'a, T> + IsShapeIndep> From<(&'a D, (usize, usize), Vec<T>)>
    for Matrix<'a, T, D>
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
#[cfg(not(feature = "no-std"))]
// FIXME: In this case, GraphReturn acts as an "IsDynamic" trait, as GraphReturn is not implemented for Stack
impl<'a, T: Copy, D: Alloc<'a, T> + IsShapeIndep> From<(&'a D, usize, usize, Vec<T>)>
    for Matrix<'a, T, D>
{
    fn from(dims_slice: (&'a D, usize, usize, Vec<T>)) -> Self {
        let data = Buffer::from((dims_slice.0, dims_slice.3));
        Matrix {
            data,
            dims: (dims_slice.1, dims_slice.2),
        }
    }
}

// FIXME: In this case, GraphReturn acts as an "IsDynamic" trait, as GraphReturn is not implemented for Stack
impl<'a, T: Copy, D: Alloc<'a, T> + IsShapeIndep> From<(&'a D, (usize, usize), &[T])>
    for Matrix<'a, T, D>
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
// FIXME: In this case, GraphReturn acts as an "IsDynamic" trait, as GraphReturn is not implemented for Stack
impl<'a, T: Copy, D: Alloc<'a, T> + IsShapeIndep> From<(&'a D, usize, usize, &[T])>
    for Matrix<'a, T, D>
{
    fn from(dims_slice: (&'a D, usize, usize, &[T])) -> Self {
        let data = Buffer::from((dims_slice.0, dims_slice.3));
        Matrix {
            data,
            dims: (dims_slice.1, dims_slice.2),
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl<'a, T: Copy, D: Alloc<'a, T> + IsShapeIndep> From<(&'a D, (usize, usize), &Vec<T>)>
    for Matrix<'a, T, D>
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

impl<'a, T, D, S: Shape> Add<Self> for &Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
{
    type Output = Matrix<'a, T, D, S>;

    fn add(self, rhs: Self) -> Self::Output {
        self.device().add(self, rhs)
    }
}

impl<'a, T, D, S: Shape> Add<Self> for Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
{
    type Output = Matrix<'a, T, D, S>;

    fn add(self, rhs: Self) -> Self::Output {
        self.device().add(&self, &rhs)
    }
}

impl<'a, T, D, S: Shape> Add<&Self> for Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
{
    type Output = Matrix<'a, T, D, S>;

    fn add(self, rhs: &Self) -> Self::Output {
        self.device().add(&self, rhs)
    }
}

impl<'a, T, D, S: Shape> Add<Matrix<'a, T, D, S>> for &Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
{
    type Output = Matrix<'a, T, D, S>;

    fn add(self, rhs: Matrix<T, D, S>) -> Self::Output {
        self.device().add(self, &rhs)
    }
}

impl<'a, T, D, S: Shape> Add<T> for &Matrix<'a, T, D, S>
where
    D: AdditionalOps<T, S>,
{
    type Output = Matrix<'a, T, D, S>;

    fn add(self, rhs: T) -> Self::Output {
        self.adds(rhs)
    }
}

impl<'a, T, D> Add<T> for Matrix<'a, T, D>
where
    D: AdditionalOps<T>,
{
    type Output = Matrix<'a, T, D>;

    fn add(self, rhs: T) -> Self::Output {
        self.adds(rhs)
    }
}

//-------------Sub-------------

impl<'a, T, D, S> Sub<Self> for &Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
    S: Shape,
{
    type Output = Matrix<'a, T, D, S>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.device().sub(self, rhs)
    }
}

impl<'a, T, D, S> Sub<Self> for Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
    S: Shape,
{
    type Output = Matrix<'a, T, D, S>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.device().sub(&self, &rhs)
    }
}

impl<'a, T, D, S> Sub<&Self> for Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
    S: Shape,
{
    type Output = Matrix<'a, T, D, S>;

    fn sub(self, rhs: &Self) -> Self::Output {
        self.device().sub(&self, rhs)
    }
}

impl<'a, T, D, S> Sub<Matrix<'a, T, D, S>> for &Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
    S: Shape,
{
    type Output = Matrix<'a, T, D, S>;

    fn sub(self, rhs: Matrix<T, D, S>) -> Self::Output {
        self.device().sub(self, &rhs)
    }
}

impl<'a, T, D, S> Sub<T> for &Matrix<'a, T, D, S>
where
    S: Shape,
    D: AdditionalOps<T, S>,
{
    type Output = Matrix<'a, T, D, S>;

    fn sub(self, rhs: T) -> Self::Output {
        self.subs(rhs);
        todo!()
        //self.subs
    }
}

impl<'a, T> Sub<T> for Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn sub(self, _rhs: T) -> Self::Output {
        todo!()
        //self.subs(rhs)
    }
}

//-------------Mul-------------

impl<'a, T, D, S: Shape> Mul<Self> for &Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
{
    type Output = Matrix<'a, T, D, S>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.device().mul(self, rhs)
    }
}

impl<'a, T, D, S: Shape> Mul<Self> for Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
{
    type Output = Matrix<'a, T, D, S>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.device().mul(&self, &rhs)
    }
}

impl<'a, T, D, S: Shape> Mul<&Self> for Matrix<'a, T, D, S>
where
    D: BaseOps<T, S>,
{
    type Output = Matrix<'a, T, D, S>;

    fn mul(self, rhs: &Self) -> Self::Output {
        self.device().mul(&self, rhs)
    }
}

impl<'a, T, S: Shape, D: AdditionalOps<T, S>> Mul<T> for Matrix<'a, T, D, S> {
    type Output = Matrix<'a, T, D, S>;

    fn mul(self, rhs: T) -> Self::Output {
        self.muls(rhs)
    }
}

impl<'a, T: Copy, S: Shape, D: AdditionalOps<T, S>> Mul<&T> for Matrix<'a, T, D, S> {
    type Output = Matrix<'a, T, D, S>;

    fn mul(self, rhs: &T) -> Self::Output {
        self.muls(*rhs)
    }
}

impl<'a, T, S: Shape, D: AdditionalOps<T, S>> Mul<T> for &Matrix<'a, T, D, S> {
    type Output = Matrix<'a, T, D, S>;

    fn mul(self, rhs: T) -> Self::Output {
        self.muls(rhs)
    }
}

// div

impl<'a, T, S: Shape, D: BaseOps<T, S>> Div<Self> for &Matrix<'a, T, D, S> {
    type Output = Matrix<'a, T, D, S>;

    fn div(self, rhs: Self) -> Self::Output {
        self.device().div(self, rhs)
    }
}

impl<'a, T, S: Shape, D: AdditionalOps<T, S>> Div<T> for Matrix<'a, T, D, S> {
    type Output = Matrix<'a, T, D, S>;

    fn div(self, rhs: T) -> Self::Output {
        self.divs(rhs)
    }
}

impl<'a, T, S: Shape, D: AdditionalOps<T, S>> Div<T> for &Matrix<'a, T, D, S> {
    type Output = Matrix<'a, T, D, S>;

    fn div(self, rhs: T) -> Self::Output {
        self.divs(rhs)
    }
}

impl<T, D, S: Shape> AddAssign<&Self> for Matrix<'_, T, D, S>
where
    D: AssignOps<T, S, D>,
{
    fn add_assign(&mut self, rhs: &Self) {
        rhs.device().add_assign(self, rhs)
    }
}

impl<T, D, S: Shape> AddAssign<Self> for Matrix<'_, T, D, S>
where
    D: AssignOps<T, S, D>,
{
    fn add_assign(&mut self, rhs: Self) {
        rhs.device().add_assign(self, &rhs)
    }
}

impl<T, D, S: Shape> MulAssign<&Self> for Matrix<'_, T, D, S>
where
    D: AssignOps<T, S, D>,
{
    fn mul_assign(&mut self, rhs: &Self) {
        rhs.device().mul_assign(self, rhs)
    }
}

impl<T, D, S: Shape> MulAssign<Self> for Matrix<'_, T, D, S>
where
    D: AssignOps<T, S, D>,
{
    fn mul_assign(&mut self, rhs: Self) {
        rhs.device().mul_assign(self, &rhs)
    }
}

impl<T, D, S: Shape> SubAssign<&Matrix<'_, T, D, S>> for &mut Matrix<'_, T, D, S>
where
    D: AssignOps<T, S, D>,
{
    fn sub_assign(&mut self, rhs: &Matrix<T, D, S>) {
        rhs.device().sub_assign(self, rhs)
    }
}

impl<T, D, S: Shape> SubAssign<Matrix<'_, T, D, S>> for &mut Matrix<'_, T, D, S>
where
    D: AssignOps<T, S, D>,
{
    fn sub_assign(&mut self, rhs: Matrix<T, D, S>) {
        rhs.device().sub_assign(self, &rhs)
    }
}

impl<T, D, S: Shape> SubAssign<&Self> for Matrix<'_, T, D, S>
where
    D: AssignOps<T, S, D>,
{
    fn sub_assign(&mut self, rhs: &Self) {
        rhs.device().sub_assign(self, rhs)
    }
}

impl<T, D, S: Shape> SubAssign<Self> for Matrix<'_, T, D, S>
where
    D: AssignOps<T, S, D>,
{
    fn sub_assign(&mut self, rhs: Self) {
        rhs.device().sub_assign(self, &rhs)
    }
}

#[cfg(not(feature = "no-std"))]
impl<'a, T: Default + Copy + core::fmt::Debug, D: Read<T, D>> core::fmt::Debug for Matrix<'a, T, D>
where
    D: Read<T, D> + 'a,
    //for<'b> <D as Read<T, D>>::Read<'b>: Iterator,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let data = self.read_to_vec();

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

#[cfg(feature = "stack")]
impl<'a, T, const N: usize> From<(&custos::Stack, usize, usize, [T; N])>
    for Matrix<'a, T, custos::Stack, custos::Dim1<N>>
{
    fn from((_, rows, cols, array): (&custos::Stack, usize, usize, [T; N])) -> Self {
        let data = Buffer::from((&custos::Stack, array));
        Matrix {
            data,
            dims: (rows, cols),
        }
    }
}

#[cfg(feature = "stack")]
impl<'a, T: Copy + Default, const A: usize, const B: usize, const N: usize>
    From<(&custos::Stack, usize, usize, [T; N])>
    for Matrix<'a, T, custos::Stack, custos::Dim2<A, B>>
{
    fn from((_, rows, cols, array): (&custos::Stack, usize, usize, [T; N])) -> Self {
        let data = Buffer::from((&custos::Stack, array));
        Matrix {
            data,
            dims: (rows, cols),
        }
    }
}

/*impl<'a, T, D: IsShapeIndep, S: Shape> From<(&D, usize, usize, [T; N])> Matrix<T, D, S> {

}*/

#[cfg(test)]
mod tests {
    use crate::Matrix;

    #[cfg(feature = "stack")]
    #[cfg(not(feature = "no-std"))]
    #[test]
    fn test_run() {
        let device = custos::CPU::new();

        let a = Matrix::from((&device, 1, 1000, [1; 1000]));
        let b = Matrix::from((&device, 1, 1000, [7; 1000]));

        loop {
            let out = &a + &b;
            assert_eq!(out.as_slice(), &[8; 1000]);
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_to_dims() {
        use custos::{Dim1, ToDim, CPU};

        let device = CPU::new();
        let a = Matrix::from((&device, 1, 1000, [1; 1000]));
        ToDim::<i32, (), Dim1<1000>>::to_dim(&device, a.data.ptr);
        //let b: custos::cpu::CPUPtr<i32> = a.device().to_dim(a.ptr);
    }
}
