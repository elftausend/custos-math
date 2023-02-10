#[cfg(feature = "opencl")]
use crate::{
    cl_diagflat,
    ops::{cl_to_cpu_lr, cl_to_cpu_s},
};
use crate::{
    matrix_multiply::MatrixMultiply, ColOp, FnsOps, Matrix, MaxOps, SumOverOps,
    TransposeOp,
};
use custos::{number::Float, range, Device, GenericBlas, CPU};
#[cfg(feature = "opencl")]
use custos::{CDatatype, OpenCL};

#[cfg(feature = "cuda")]
use crate::{cu_to_cpu_lr, cu_to_cpu_s};
#[cfg(feature = "cuda")]
use custos::CUDA;

impl<'a, T, D: SoftmaxOps<T>> Matrix<'a, T, D> {
    pub fn softmax(&self) -> Matrix<'a, T, D> {
        self.device().softmax(self)
    }
    pub fn softmax_grad(&self, activated: &Matrix<T, D>) -> Matrix<'a, T, D> {
        self.device().softmax_grad(activated, self)
    }
}

pub trait SoftmaxOps<T, D: Device = Self>: Device {
    fn softmax(&self, inputs: &Matrix<T, D>) -> Matrix<T, Self>;
    fn softmax_grad(&self, activated: &Matrix<T, D>, grads: &Matrix<T, D>) -> Matrix<T, Self>;
}

#[cfg(feature = "cpu")]
impl<T: Float + GenericBlas + MatrixMultiply> SoftmaxOps<T> for CPU
where
    CPU: ColOp<T>,
{
    fn softmax(&self, inputs: &Matrix<T>) -> Matrix<T> {
        let exp = self.exp(&self.sub_col(inputs, &self.max_cols(inputs)));
        self.div_col(&exp, &self.sum_cols(&exp))
    }

    #[cfg(not(feature = "safe"))]
    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T> {
        use custos::Cache;

        use crate::{BaseOps, Gemm};

        let mut data: Matrix<T> = (Cache::get(self, grads.len(), ()), grads.dims()).into();

        let rows = grads.rows();
        let cols = grads.cols();

        for idx in range(rows - 1) {
            let index = idx * cols;

            let single_out = Matrix::from((
                self,
                (&activated[index..index + cols]).as_ptr() as *mut T,
                (cols, 1),
            ));
            let single_grad = Matrix::from((
                self,
                (&grads[index..index + cols]).as_ptr() as *mut T,
                (cols, 1),
            ));

            let diagflat = single_out.diagflat();

            // cols 1 x 1 cols
            let jacobian_matrix =
                self.sub(&diagflat, &self.gemm(&single_out, &single_out.T::<()>()));

            //GenericBlas::gemm();
            let res: Matrix<T> = jacobian_matrix.gemm(&single_grad);

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
impl<T: GenericBlas + MatrixMultiply + Float> SoftmaxOps<T> for CUDA {
    fn softmax(&self, inputs: &Matrix<T, Self>) -> Matrix<T, Self> {
        cu_to_cpu_s(self, inputs, |cpu, x| cpu.softmax(&x))
    }

    fn softmax_grad(
        &self,
        activated: &Matrix<T, Self>,
        grads: &Matrix<T, Self>,
    ) -> Matrix<T, Self> {
        cu_to_cpu_lr(self, activated, grads, |cpu, activated, grads| {
            cpu.softmax_grad(activated, grads)
        })
    }
}

#[cfg(feature = "opencl")]
// TODO: Softmax running on the opencl device
impl<T: GenericBlas + MatrixMultiply + Float> SoftmaxOps<T> for OpenCL {
    fn softmax(&self, inputs: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_to_cpu_s(self, inputs, |device, inputs| device.softmax(inputs))
    }

    fn softmax_grad(
        &self,
        activated: &Matrix<T, Self>,
        grads: &Matrix<T, Self>,
    ) -> Matrix<T, Self> {
        cl_to_cpu_lr(self, activated, grads, |device, activated, grads| {
            device.softmax_grad(activated, grads)
        })
    }
}

#[cfg(feature = "opencl")]
pub fn cl_softmax<'a, T: CDatatype>(
    device: &'a OpenCL,
    mut activated: Matrix<T, OpenCL>,
    grads: &Matrix<T, OpenCL>,
) -> custos::Result<Matrix<'a, T, OpenCL>> {
    use crate::{cl_tew, Gemm, SumOverOps};

    let rows = grads.rows();
    let cols = grads.cols();

    let diag = cl_diagflat(device, &activated, activated.rows(), activated.cols())?;

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
    let res = device.gemm(grads, &jacobian);
    Ok(res)
}
