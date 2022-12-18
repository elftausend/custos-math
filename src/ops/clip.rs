use custos::{cache::Cache, number::Number, CDatatype, Device, MainMemory, CPU};

#[cfg(feature = "opencl")]
use custos::OpenCL;

use crate::Matrix;
#[cfg(feature = "cuda")]
use custos::{cuda::launch_kernel1d, Buffer, CudaDevice};

impl<'a, T: CDatatype, D: ClipOp<T>> Matrix<'a, T, D> {
    pub fn clip(&self, min: T, max: T) -> Matrix<T, D> {
        self.device().clip(self, min, max)
    }
}

pub trait ClipOp<T, D: Device = Self>: Device {
    fn clip(&self, x: &Matrix<T, D>, min: T, max: T) -> Matrix<T, Self>;
}

impl<T: Number, D: MainMemory> ClipOp<T, D> for CPU {
    fn clip(&self, x: &Matrix<T, D>, min: T, max: T) -> Matrix<T> {
        let mut y = Cache::get::<T, 0>(self, x.size(), x.node.idx);
        let y_slice = y.as_mut_slice();

        for (idx, value) in x.as_slice().iter().enumerate() {
            if *value < min {
                y_slice[idx] = min;
            } else if *value > max {
                y_slice[idx] = max;
            } else {
                y_slice[idx] = *value;
            }
        }
        (y, x.dims()).into()
    }
}

#[cfg(feature = "opencl")]
fn cl_clip<'a, T: CDatatype>(
    device: &'a OpenCL,
    x: &Matrix<T>,
    min: T,
    max: T,
) -> custos::Result<Matrix<'a, T>> {
    use custos::opencl::enqueue_kernel;

    let src = format!(
        "
        #define MIN {min}
        #define MAX {max}
        __kernel void clip(__global const {datatype}* input, __global {datatype}* output) {{

            size_t id = get_global_id(0);
            if (input[id] < MIN) {{
                output[id] = MIN;
            }} else if (input[id] > MAX) {{
                output[id] = MAX;
            }} else {{
                output[id] = input[id];
            }} 
        }}
    ",
        datatype = T::as_c_type_str()
    );

    let out = Cache::get::<T, 0>(device, x.size(), x.node.idx);
    enqueue_kernel(device, &src, [x.size(), 0, 0], None, &[x, &out])?;
    Ok((out, x.dims()).into())
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> ClipOp<T> for OpenCL {
    fn clip(&self, x: &Matrix<T>, min: T, max: T) -> Matrix<T> {
        cl_clip(self, x, min, max).unwrap()
    }
}

#[cfg(feature = "cuda")]
pub fn cu_clip<'a, T: CDatatype>(
    device: &'a CudaDevice,
    x: &Buffer<T>,
    min: T,
    max: T,
) -> custos::Result<Buffer<'a, T>> {
    let src = format!(
        r#"extern "C" __global__ void clip({datatype}* lhs, {datatype} min, {datatype} max, {datatype}* out, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    {datatype} value = lhs[idx];
                    if (value > max) {{
                        out[idx] = max;
                    }} else if (value < min) {{
                        out[idx] = min;
                    }} else {{
                        out[idx] = value;
                    }}
                }}
              
            }}
    "#,
        datatype = T::as_c_type_str()
    );

    let out = Cache::get::<T, _>(device, x.len(), x.node.idx);
    launch_kernel1d(
        x.len(),
        device,
        &src,
        "clip",
        &[x, &min, &max, &out, &x.len],
    )?;
    Ok(out)
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> ClipOp<T> for CudaDevice {
    fn clip(&self, x: &Matrix<T>, min: T, max: T) -> Matrix<T> {
        let buf = cu_clip(self, x, min, max).unwrap();
        (buf, x.dims()).into()
    }
}
