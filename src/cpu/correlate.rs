use custos::number::Number;

pub fn correlate_valid_mut<T: Number>(lhs_slice: &[T], lhs_dims: (usize, usize), 
                                kernel_slice: &[T], rhs_dims: (usize, usize), out: &mut [T])                             
{
    let (lhs_rows, lhs_cols) = lhs_dims;
    let (rhs_rows, rhs_cols) = rhs_dims;
    
    let (out_rows, out_cols) = (lhs_rows-rhs_rows+1, lhs_cols-rhs_cols+1);

    //loop for row-axis (y)
    //moves multiplication 1 down
    for y in 0..out_rows {
        //loop for col-axis (x)
        //moves multiplication 1 to the right
        for x in 0..out_cols {
            let mut sum = T::default();
            //repeat 'kernel rows' (rhs_rows) times to use move through all kernel rows
            for idx in 0..rhs_rows {
                
                let index = idx*lhs_cols +x + y*lhs_cols;
                let lhs_kernel_row = &lhs_slice[index..index+rhs_cols];
                
                let index = idx*rhs_cols;
                let kernel_row = &kernel_slice[index..index+rhs_cols];
                
                for (i, value) in lhs_kernel_row.iter().enumerate() {
                    sum += *value*kernel_row[i];
                }

            }
            // y * final_cols + x
            out[y*out_cols+x] = sum;
        }
    }
}
