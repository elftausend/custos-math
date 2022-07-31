use crate::{cached, ColOp, DiagflatOp, FnsOps, Matrix, MaxOps, SumOps, TransposeOp};
#[cfg(feature = "opencl")]
use crate::{
    cl_diagflat,
    ops::{cl_to_cpu_lr, cl_to_cpu_s},
};
use custos::{get_device, number::Float, range, GenericBlas, CPU};
#[cfg(feature = "opencl")]
use custos::{CDatatype, CLDevice};

#[cfg(feature = "cuda")]
use crate::{cu_to_cpu_lr, cu_to_cpu_s};
#[cfg(feature = "cuda")]
use custos::CudaDevice;

pub trait Softmax<T> {
    fn softmax(&self) -> Matrix<T>;
    fn softmax_grad(&self, activated: &Matrix<T>) -> Matrix<T>;
}

impl<'a, T: GenericBlas> Matrix<'a, T> {
    pub fn softmax(&self) -> Matrix<'a, T> {
        let device = get_device!(self.device(), SoftmaxOps<T>);
        device.softmax(self)
    }
    pub fn softmax_grad(&self, activated: &Matrix<T>) -> Matrix<'a, T> {
        let device = get_device!(self.device(), SoftmaxOps<T>);
        device.softmax_grad(activated, self)
    }
}

pub trait SoftmaxOps<T> {
    fn softmax(&self, inputs: &Matrix<T>) -> Matrix<T>;
    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T>;
}

impl<T: Float + GenericBlas> SoftmaxOps<T> for CPU {
    fn softmax(&self, inputs: &Matrix<T>) -> Matrix<T> {
        let exp = self.exp(&self.sub_col(inputs, &self.max_cols(inputs)));
        self.div_col(&exp, &self.sum_cols(&exp))
    }

    #[cfg(not(feature = "safe"))]
    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T> {
        use crate::{BaseOps, Gemm};

        let mut data = cached(self, grads.dims());

        let rows = grads.rows();
        let cols = grads.cols();

        for idx in range(rows - 1) {
            let index = idx * cols;

            let single_out = Matrix::from((
                (&activated[index..index + cols]).as_ptr() as *mut T,
                (cols, 1),
            ));
            let single_grad =
                Matrix::from(((&grads[index..index + cols]).as_ptr() as *mut T, (cols, 1)));

            let diagflat = self.diagflat(&single_out);

            // cols 1 x 1 cols
            let jacobian_matrix = self.sub(
                &diagflat,
                &self.gemm(&single_out, &self.transpose(&single_out)),
            );

            let res = self.gemm(&jacobian_matrix, &single_grad);

            let data_row = &mut data[index..index + cols];
            data_row.copy_from_slice(&res);
        }
        data
    }

    #[cfg(feature = "safe")]
    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T> {
        use crate::{BaseOps, Gemm};

        let device = CPU::new();
        let mut data = cached(self, grads.dims());

        let rows = grads.rows();
        let cols = grads.cols();

        for idx in range(rows - 1) {
            let index = idx * cols;

            let single_out =
                Matrix::from((&device, (cols, 1), &activated[index..index + cols].to_vec()));

            let single_grad =
                Matrix::from((&device, (cols, 1), &grads[index..index + cols].to_vec()));

            let diagflat = self.diagflat(&single_out);

            let jacobian_matrix = self.sub(
                &diagflat,
                &self.gemm(&single_out, &self.transpose(&single_out)),
            );
            //cols cols x cols 1
            let res = self.gemm(&jacobian_matrix, &single_grad);

            let data_row = &mut data[index..index + cols];
            data_row.copy_from_slice(res.as_slice());
        }
        data
    }
}

#[cfg(feature = "cuda")]
impl<T: Default + Copy + GenericBlas> SoftmaxOps<T> for CudaDevice {
    fn softmax(&self, inputs: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_s(self, inputs, |cpu, x| cpu.softmax(&x))
    }

    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_lr(self, activated, grads, |cpu, activated, grads| {
            cpu.softmax_grad(activated, grads)
        })
    }
}

#[cfg(feature = "opencl")]
// TODO: Softmax running on the opencl device
impl<T: GenericBlas + Float> SoftmaxOps<T> for CLDevice {
    fn softmax(&self, inputs: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_s(self, inputs, |device, inputs| device.softmax(&inputs))
    }

    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_lr(self, activated, grads, |device, activated, grads| {
            device.softmax_grad(activated, grads)
        })
    }
}

#[cfg(feature = "opencl")]
pub fn cl_softmax<'a, T: CDatatype>(
    device: &'a CLDevice,
    mut activated: Matrix<T>,
    grads: &Matrix<T>,
) -> custos::Result<Matrix<'a, T>> {
    use crate::{cl_tew, Gemm};

    let rows = grads.rows();
    let cols = grads.cols();

    let diag = cl_diagflat(device, &activated)?;

    //println!("diag: {diag:?}");
    activated.reshape((cols, rows));

    //cols rows x rows cols

    let jacobian = cl_tew(
        device,
        &diag,
        &device.gemm(&activated, &device.transpose(&activated)),
        "-",
    )?;

    println!("jacobian: {jacobian:?}");

    let jacobian = (jacobian, rows, cols * cols).into();
    let mut jacobian = device.sum_rows(&jacobian);
    jacobian.reshape((cols, cols));

    // rows cols x cols cols
    let res = device.gemm(&grads, &jacobian);
    Ok(res)
}
