use core::ops::{Bound, Range, RangeBounds};

use custos::{impl_stack, number::Number, CopySlice, Device, Shape, CPU};

#[cfg(feature = "stack")]
use custos::Stack;

use crate::Matrix;

pub trait SliceOps<T, S: Shape = (), D: Device = Self>: Device + CopySlice<T> {
    /// Slice
    /// # Examples
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Read};
    /// use custos_math::{Matrix, SliceOps};
    ///
    /// let device = CPU::new();
    /// let x = Matrix::from((&device, 2, 3, [
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// ]));
    ///
    /// assert_eq!(device.slice(&x, ..1, ..).read(), vec![1, 2, 3]);
    /// //assert_eq!(device.slice(&x, .., 1..3).read(), vec![2, 3, 5, 6]);
    /// ```

    fn slice<R, C>(&self, source: &Matrix<T, D, S>, rows: R, cols: C) -> Matrix<T, D, S>
    where
        R: RangeBounds<usize>,
        C: RangeBounds<usize>;
}

#[impl_stack]
impl<T: Number> SliceOps<T> for CPU
where
    Self: CopySlice<T>,
{
    fn slice<R, C>(&self, source: &Matrix<T, CPU, ()>, rows: R, cols: C) -> Matrix<T, CPU, ()>
    where
        R: RangeBounds<usize>,
        C: RangeBounds<usize>,
    {
        let rows = to_range(rows, source.rows());
        let cols = to_range(cols, source.cols());

        let num_rows = rows.end - rows.start;
        let num_cols = cols.end - cols.start;

        if num_cols == source.cols() {
            let offset = rows.start * num_cols;
            let size = num_rows * num_cols;
            let buffer = self.copy_slice(source.as_buf(), offset..(offset + size));
            (buffer, (num_rows, num_cols)).into()
        } else {
            todo!()
        }
    }
}

#[inline]
fn to_range<B: RangeBounds<usize>>(bounds: B, len: usize) -> Range<usize> {
    let start = match bounds.start_bound() {
        Bound::Included(start) => *start,
        Bound::Excluded(start) => start + 1,
        Bound::Unbounded => 0,
    };

    let end = match bounds.end_bound() {
        Bound::Excluded(end) => *end,
        Bound::Included(end) => end + 1,
        Bound::Unbounded => len,
    };

    start..end
}
