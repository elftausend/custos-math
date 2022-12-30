use custos::{number::Number, CDatatype, Device, MainMemory, CPU};

#[cfg(feature = "opencl")]
use custos::OpenCL;

use crate::Matrix;
#[cfg(feature = "cuda")]
use custos::{cuda::launch_kernel1d, Buffer, CUDA};

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
        let mut out = self.retrieve(x.size(), x.node.idx);

        for (idx, value) in x.iter().enumerate() {
            if *value < min {
                out[idx] = min;
            } else if *value > max {
                out[idx] = max;
            } else {
                out[idx] = *value;
            }
        }
        (out, x.dims()).into()
    }
}

#[cfg(feature = "opencl")]
fn cl_clip<'a, T: CDatatype>(
    device: &'a OpenCL,
    x: &Matrix<T, OpenCL>,
    min: T,
    max: T,
) -> custos::Result<Matrix<'a, T, OpenCL>> {
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

    let out = device.retrieve::<T, ()>(x.size(), x.node.idx);
    enqueue_kernel(device, &src, [x.size(), 0, 0], None, &[x, &out])?;
    Ok((out, x.dims()).into())
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> ClipOp<T> for OpenCL {
    fn clip(&self, x: &Matrix<T, Self>, min: T, max: T) -> Matrix<T, Self> {
        cl_clip(self, x, min, max).unwrap()
    }
}

#[cfg(feature = "cuda")]
pub fn cu_clip<'a, T: CDatatype>(
    device: &'a CUDA,
    x: &Buffer<T, CUDA>,
    min: T,
    max: T,
) -> custos::Result<Buffer<'a, T, CUDA>> {
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

    let out = Cache::get::<T, 0>(device, x.len(), x.node.idx);
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
impl<T: CDatatype> ClipOp<T> for CUDA {
    fn clip(&self, x: &Matrix<T, CUDA>, min: T, max: T) -> Matrix<T, CUDA> {
        let buf = cu_clip(self, x, min, max).unwrap();
        (buf, x.dims()).into()
    }
}
