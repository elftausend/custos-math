use custos::{
    get_device, number::Float, opencl::cl_write, Device, InternCLDevice, InternCPU, Matrix, VecRead,
};
use rand::{thread_rng, Rng, distributions::uniform::SampleUniform};

pub trait RandMatrix<T> {
    fn rand(&mut self, lo: T, hi: T);
}
impl<T: Float + SampleUniform> RandMatrix<T> for Matrix<T> {
    fn rand(&mut self, lo: T, hi: T) {
        let device = get_device!(RandOp, T).unwrap();
        device.rand(self, lo, hi)
    }
}

pub trait RandOp<T>: Device<T> {
    fn rand(&self, x: &mut Matrix<T>, lo: T, hi: T);
}

impl<T: Float + SampleUniform> RandOp<T> for InternCPU {
    fn rand(&self, x: &mut Matrix<T>, lo: T, hi: T) {
        let mut rng = thread_rng();
        for value in x.as_mut_slice() {
            *value = rng.gen_range(lo..hi);
        }
    }
}

impl<T: Float + SampleUniform> RandOp<T> for InternCLDevice {
    fn rand(&self, x: &mut Matrix<T>, lo: T, hi: T) {
        let mut rng = thread_rng();
        let mut data = self.read(x.as_buf());

        for value in data.iter_mut() {
            *value = rng.gen_range(lo..hi);
        }
        cl_write(self, x, &data);
    }
}
