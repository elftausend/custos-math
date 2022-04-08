use custos::{Matrix, cpu::{InternCPU, CPUCache}, number::Number, opencl::{InternCLDevice, KernelOptions}, Error, get_device, GenericOCL};

pub trait Clip<T> {
    fn clip(&self, min: T, max: T) -> Matrix<T>;
}

impl <T: GenericOCL>Clip<T> for Matrix<T> {
    fn clip(&self, min: T, max: T) -> Matrix<T> {
        let device = get_device!(ClipOp, T).unwrap();
        device.clip(*self, min, max)
    }
}

pub trait ClipOp<T> {
    fn clip(&self, x: Matrix<T>, min: T, max: T) -> Matrix<T>;
}

impl <T: Number>ClipOp<T> for InternCPU {
    fn clip(&self, x: Matrix<T>, min: T, max: T) -> Matrix<T> {
        let mut y = CPUCache::get::<T>(self.clone(), x.dims());
        let y_slice = y.as_cpu_slice_mut();

        for (idx, value) in x.as_cpu_slice().iter().enumerate() {
            if *value < min {
                y_slice[idx] = min;
            } else if *value > max {
                y_slice[idx] = max;
            } else {
                y_slice[idx] = *value;
            }
        }
        y
    }
}

fn ocl_clip<T: GenericOCL>(device: InternCLDevice, x: Matrix<T>, min: T, max: T) -> Result<Matrix<T>, Error> {
    let src = format!("
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
    ", datatype=T::as_ocl_type_str());

    KernelOptions::new(&device, &x, [x.size(), 0, 0], &src)
        .with_output(x.dims())
        .run()

}

impl <T: GenericOCL>ClipOp<T> for InternCLDevice {
    fn clip(&self, x: Matrix<T>, min: T, max: T) -> Matrix<T> {
        ocl_clip(self.clone(), x, min, max).unwrap()
    }
}