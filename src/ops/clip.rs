use custos::{
    cpu::{CPUCache, CPU},
    get_device,
    number::Number,
    opencl::{CLDevice, KernelOptions},
    CDatatype, Matrix, CudaDevice,
};

pub trait Clip<T> {
    fn clip(&self, min: T, max: T) -> Matrix<T>;
}

impl<T: CDatatype> Clip<T> for Matrix<T> {
    fn clip(&self, min: T, max: T) -> Matrix<T> {
        let device = get_device!(ClipOp, T).unwrap();
        device.clip(self, min, max)
    }
}

pub trait ClipOp<T> {
    fn clip(&self, x: &Matrix<T>, min: T, max: T) -> Matrix<T>;
}

impl<T: Number> ClipOp<T> for CPU {
    fn clip(&self, x: &Matrix<T>, min: T, max: T) -> Matrix<T> {
        let mut y = CPUCache::get::<T>(self, x.size());
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

fn ocl_clip<T: CDatatype>(
    device: CLDevice,
    x: &Matrix<T>,
    min: T,
    max: T,
) -> custos::Result<Matrix<T>> {
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

    let buf = KernelOptions::new(&device, x, [x.size(), 0, 0], &src)?
        .with_output(x.size())
        .run();

    // TODO: unwrap, Ok()?
    buf.map(|buf| (buf.unwrap(), x.dims()).into())

}

impl<T: CDatatype> ClipOp<T> for CLDevice {
    fn clip(&self, x: &Matrix<T>, min: T, max: T) -> Matrix<T> {
        ocl_clip(self.clone(), x, min, max).unwrap()
    }
}

impl<T> ClipOp<T> for CudaDevice {
    fn clip(&self, x: &Matrix<T>, min: T, max: T) -> Matrix<T> {
        todo!()
    }
}