use custos::{cuda::launch_kernel, prelude::CUBuffer, CDatatype, CUDA};


pub fn add_to_row_cu_2dim<T: CDatatype>(
    device: &CUDA,
    lhs: &CUBuffer<T>,
    m: usize,
    n: usize,
    rhs: &CUBuffer<T>,
    out: &mut CUBuffer<T>,
) -> custos::Result<()> {
    let src = format!(
        r#"
        extern "C" __global__ void addToRow({dtype}* lhs, {dtype}* rhs, {dtype}* out, size_t m, size_t n) {{
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (y >= m || x >= n) {{
                return;
            }}

            __shared__ {dtype} s_rhs[32];
            s_rhs[threadIdx.x] = rhs[x];

            __syncthreads();

            out[y * n + x] = lhs[y * n + x] + s_rhs[threadIdx.x];
        }} 
    "#,
        dtype = T::as_c_type_str()
    );

    launch_kernel(
        device,
        [u32::max(1, n as u32 / 32), u32::max(1, m as u32 / 32), 1],
        [32, 32, 1],
        0,
        &src,
        "addToRow",
        &[lhs, rhs, out, &m, &n],
    )
}

pub fn add_to_row_cu_in_place<T: CDatatype>(
    device: &CUDA,
    lhs: &mut CUBuffer<T>,
    m: usize,
    n: usize,
    rhs: &CUBuffer<T>,
) -> custos::Result<()> {
    let src = format!(
        r#"
        extern "C" __global__ void addToRowInPlace({dtype}* lhs, {dtype}* rhs, size_t m, size_t n) {{
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (y >= m || x >= n) {{
                return;
            }}

            __shared__ {dtype} s_rhs[32];
            s_rhs[threadIdx.x] = rhs[x];

            __syncthreads();

            lhs[y * n + x] += s_rhs[threadIdx.x];
        }} 
    "#,
        dtype = T::as_c_type_str()
    );

    launch_kernel(
        device,
        [u32::max(1, n as u32 / 32), u32::max(1, m as u32 / 32), 1],
        [32, 32, 1],
        0,
        &src,
        "addToRowInPlace",
        &[lhs, rhs, &m, &n],
    )
}

pub fn add_to_row_cu<T: CDatatype>(
    device: &CUDA,
    lhs: &CUBuffer<T>,
    m: usize,
    n: usize,
    rhs: &CUBuffer<T>,
    out: &mut CUBuffer<T>,
) -> custos::Result<()> {
    let src = format!(
        r#"
        extern "C" __global__ void addToRow({dtype}* lhs, {dtype}* rhs, {dtype}* out, size_t size) {{
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= size) {{
                return;
            }}

            int row = idx % {n};
            out[idx] = lhs[idx] + rhs[row];
        }} 
    "#,
        dtype = T::as_c_type_str()
    );

    launch_kernel(
        device,
        [((m * n) / 32) as u32+1, 1, 1],
        [32, 1, 1],
        0,
        &src,
        "addToRow",
        &[lhs, rhs, out, &lhs.len()],
    )
}

#[cfg(test)]
mod tests {
    use custos::{Buffer, CUDA};

    use crate::{add_to_row_cu, add_to_row_cu_2dim};

    #[test]
    fn test_add_to_row_cu() {
        let device = CUDA::new(0).unwrap();

        let m = 3;
        let n = 6;
        // 3x6
        #[rustfmt::skip]
        let lhs = Buffer::from((&device, [
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 0, 1, 2,
            3, 4, 5, 6, 7, 8,
        ]));

        let rhs = Buffer::from((&device, [4, 3, 5, 1, 2, 6]));
        let mut out = Buffer::new(&device, lhs.len());
        add_to_row_cu(&device, &lhs, m, n, &rhs, &mut out).unwrap();
        
        #[rustfmt::skip]
        assert_eq!(out.read(), [
            5, 5, 8, 5, 7, 12,
            11, 11, 14, 1, 3, 8,
            7, 7, 10, 7, 9, 14
        ]);
    }

    #[test]
    fn test_add_to_row_cu_2dim() {
        let device = CUDA::new(0).unwrap();

        let m = 3;
        let n = 6;
        // 3x6
        #[rustfmt::skip]
        let lhs = Buffer::from((&device, [
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 0, 1, 2,
            3, 4, 5, 6, 7, 8,
        ]));

        let rhs = Buffer::from((&device, [4, 3, 5, 1, 2, 6]));
        let mut out = Buffer::new(&device, lhs.len());
        add_to_row_cu_2dim(&device, &lhs, m, n, &rhs, &mut out).unwrap();
        
        #[rustfmt::skip]
        assert_eq!(out.read(), [
            5, 5, 8, 5, 7, 12,
            11, 11, 14, 1, 3, 8,
            7, 7, 10, 7, 9, 14
        ]);
    }
}
