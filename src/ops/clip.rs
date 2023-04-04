use custos::{impl_stack, number::Number, CDatatype, Device, MainMemory, Shape, CPU};

#[cfg(feature = "stack")]
use custos::Stack;

#[cfg(feature = "opencl")]
use custos::OpenCL;

use crate::Matrix;
#[cfg(feature = "cuda")]
use custos::{cuda::launch_kernel1d, Buffer, CUDA};

impl<'a, T, S: Shape, D: ClipOp<T, S>> Matrix<'a, T, D, S> {
    pub fn clip(&self, min: T, max: T) -> Matrix<T, D, S> {
        self.device().clip(self, min, max)
    }
}

pub trait ClipOp<T, S: Shape = (), D: Device = Self>: Device {
    fn clip(&self, x: &Matrix<T, D, S>, min: T, max: T) -> Matrix<T, Self, S>;
}

#[impl_stack]
impl<T: Number, D: MainMemory, S: Shape> ClipOp<T, S, D> for CPU {
    fn clip(&self, x: &Matrix<T, D, S>, min: T, max: T) -> Matrix<T, Self, S> {
        let mut out = self.retrieve(x.size(), x.as_buf());
        let out_slice = &mut out[..];

        for (idx, value) in x.iter().enumerate() {
            if *value < min {
                out_slice[idx] = min;
            } else if *value > max {
                out_slice[idx] = max;
            } else {
                out_slice[idx] = *value;
            }
        }
        (out, x.dims()).into()
    }
}

#[cfg(feature = "opencl")]
fn cl_clip<'a, T: CDatatype + Number>(
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

    let out = device.retrieve::<T, ()>(x.size(), x.as_buf());
    enqueue_kernel(device, &src, [x.size(), 0, 0], None, &[x, &out])?;
    Ok((out, x.dims()).into())
}

#[cfg(feature = "opencl")]
impl<T: CDatatype + Number> ClipOp<T> for OpenCL {
    fn clip(&self, x: &Matrix<T, Self>, min: T, max: T) -> Matrix<T, Self> {
        cl_clip(self, x, min, max).unwrap()
    }
}

#[cfg(feature = "cuda")]
pub fn cu_clip<'a, T: CDatatype + Number>(
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

    let out = device.retrieve::<T, ()>(x.len(), x);
    launch_kernel1d(
        x.len(),
        device,
        &src,
        "clip",
        &[x, &min, &max, &out, &x.len()],
    )?;
    Ok(out)
}

#[cfg(feature = "cuda")]
impl<T: CDatatype + Number> ClipOp<T> for CUDA {
    fn clip(&self, x: &Matrix<T, CUDA>, min: T, max: T) -> Matrix<T, CUDA> {
        let buf = cu_clip(self, x, min, max).unwrap();
        (buf, x.dims()).into()
    }
}
