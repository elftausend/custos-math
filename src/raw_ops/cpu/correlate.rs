use custos::number::Number;

pub fn correlate_valid_mut<T: Number>(
    lhs_slice: &[T],
    lhs_dims: (usize, usize),
    kernel_slice: &[T],
    kernel_dims: (usize, usize),
    out: &mut [T],
) {
    let (lhs_rows, lhs_cols) = lhs_dims;
    let (kernel_rows, kernel_cols) = kernel_dims;

    let (out_rows, out_cols) = (lhs_rows - kernel_rows + 1, lhs_cols - kernel_cols + 1);

    //loop for row-axis (y)
    //moves multiplication 1 down
    for y in 0..out_rows {
        //loop for col-axis (x)
        //moves multiplication 1 to the right
        for x in 0..out_cols {
            let mut sum = T::default();
            //repeat kernel rows times to use move through all kernel rows
            for idx in 0..kernel_rows {
                let index = idx * lhs_cols + x + y * lhs_cols;
                let lhs_kernel_row = &lhs_slice[index..index + kernel_cols];

                let index = idx * kernel_cols;
                let kernel_row = &kernel_slice[index..index + kernel_cols];

                for (i, value) in lhs_kernel_row.iter().enumerate() {
                    sum += *value * kernel_row[i];
                }
            }
            // y * final_cols + x
            out[y * out_cols + x] = sum;
        }
    }
}

#[cfg(not(feature = "no-std"))]
pub fn add_full_padding<T: Number>(
    lhs: &[T],
    lhs_dims: (usize, usize),
    kernel_dims: (usize, usize),
) -> (Vec<T>, usize, usize) {
    let (lhs_rows, lhs_cols) = lhs_dims;
    let (kernel_rows, kernel_cols) = kernel_dims;

    let (row_adding, col_adding) = ((kernel_rows - 1) * 2, (kernel_cols - 1) * 2);
    let (out_rows, out_cols) = (lhs_rows + row_adding, lhs_cols + col_adding);

    let mut out = vec![T::default(); out_rows * out_cols];

    for row in 0..lhs_rows {
        let idx = row * lhs_cols;
        let lhs_row = &lhs[idx..idx + lhs_cols];

        let index = (row + (kernel_rows - 1)) * (out_cols) + (kernel_cols - 1);
        let out_row = &mut out[index..index + out_cols];

        for (idx, value) in lhs_row.iter().enumerate() {
            out_row[idx] = *value;
        }
    }
    (out, out_rows, out_cols)
}

#[cfg(not(feature = "no-std"))]
pub fn rot_kernel<T: Number>(kernel: &[T], kernel_shape: (usize, usize)) -> Vec<T> {
    let (kernel_rows, kernel_cols) = kernel_shape;
    let mut rotated = vec![T::default(); kernel.len()];

    for (idx_rev, idx) in (0..kernel_rows).rev().zip(0..kernel_rows) {
        let row_idx = idx_rev * kernel_cols;
        let row = &kernel[row_idx..row_idx + kernel_cols];

        for (i, value) in row.iter().rev().enumerate() {
            rotated[idx * kernel_cols + i] = *value;
        }
    }
    rotated
}
