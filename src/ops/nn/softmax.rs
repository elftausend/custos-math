use crate::{
    cached,
    ops::{cl_to_cpu_lr, cl_to_cpu_s},
    ColOp, DiagflatOp, FnsOps, Mat, MaxOps, SumOps, TransposeOp, cl_diagflat,
};
use custos::{
    GenericBlas, get_device, number::Float, range, BaseOps, Gemm, CDatatype, CLDevice,
    CPU, Matrix, opencl::cl_tew,
};

pub trait Softmax<T> {
    fn softmax(&self) -> Matrix<T>;
    fn softmax_grad(&self, activated: Matrix<T>) -> Matrix<T>;
}

impl<T: CDatatype + GenericBlas + Float, L: Mat<T>> Softmax<T> for L {
    fn softmax(&self) -> Matrix<T> {
        let device = get_device!(SoftmaxOps, T).unwrap();
        device.softmax(self.as_mat())
    }

    fn softmax_grad(&self, activated: Matrix<T>) -> Matrix<T> {
        let device = get_device!(SoftmaxOps, T).unwrap();
        device.softmax_grad(activated.as_mat(), self.as_mat())
    }
}

pub trait SoftmaxOps<T> {
    fn softmax(&self, inputs: &Matrix<T>) -> Matrix<T>;
    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T>;
}

impl<T: Float + GenericBlas> SoftmaxOps<T> for CPU {
    fn softmax(&self, inputs: &Matrix<T>) -> Matrix<T> {
        let exp = self.exp(&self.sub_col(&inputs, &self.max_cols(inputs)));
        self.div_col(&exp, &self.sum_cols(&exp))
    }

    #[cfg(not(feature = "safe"))]
    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T> {
        let mut data = cached(self, grads.dims());

        let rows = grads.rows();
        let cols = grads.cols();

        for idx in range(rows - 1) {
            let index = idx * cols;

            let single_out = Matrix::from((
                (&activated[index..index + cols]).as_ptr() as *mut T,
                (cols, 1),
            ));
            let single_grad = Matrix::from((
                (&grads[index..index + cols]).as_ptr() as *mut T,
                (cols, 1),
            ));

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
        use custos::CPU;
        let device = CPU::new();
        let mut data = cached(self, grads.dims());

        let rows = grads.rows();
        let cols = grads.cols();

        for idx in range(rows - 1) {
            let index = idx * cols;

            let single_out = Matrix::from((
                &device,
                (cols, 1),
                &activated[index..index + cols].to_vec(),
            ));

            let single_grad =
                Matrix::from((&device, (cols, 1), &grad[index..index + cols].to_vec()));
        
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

// TODO: Softmax running on the opencl device
impl<T: CDatatype + GenericBlas + Float> SoftmaxOps<T> for CLDevice {
    fn softmax(&self, inputs: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_s(self, inputs, |device, inputs| device.softmax(&inputs))
    }

    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_lr(self, activated, &grads, |device, activated, grads| {
            device.softmax_grad(activated, &grads)
        })
    }
}

pub fn cl_softmax<T: CDatatype>(device: &CLDevice, mut activated: Matrix<T>, grads: &Matrix<T>) -> custos::Result<Matrix<T>> {
    let rows = grads.rows();
    let cols = grads.cols();

    let diag = cl_diagflat(device, &activated)?;

    //println!("diag: {diag:?}");
    activated.reshape((cols, rows));

    //cols rows x rows cols

    let jacobian = cl_tew(
        device, 
        &diag, 
        &device.gemm(&activated, &device.transpose(&activated)), "-"
    )?;

    println!("jacobian: {jacobian:?}");

    let jacobian = (jacobian, rows, cols*cols).into();
    let mut jacobian = device.sum_rows(&jacobian);
    jacobian.reshape((cols, cols));
    
    // rows cols x cols cols
    let res = device.gemm(&grads, &jacobian);
    Ok(res)
}