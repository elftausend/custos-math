use custos::{Matrix, InternCPU, number::Float, range, Gemm, cpu::TBlas, BaseOps, InternCLDevice, get_device, GenericOCL};
use crate::{SumOps, MaxOps, ColOp, FnsOps, DiagflatOp, TransposeOp, ops::{switch_to_cpu_help_s, switch_to_cpu_help_lr}, cached, Mat};

pub trait Softmax<T> {
    fn softmax(&self) -> Matrix<T>;
    fn softmax_grad(&self, activated: Matrix<T>) -> Matrix<T>;
}

impl <T: GenericOCL+TBlas+Float, L: Mat<T>>Softmax<T> for L {
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

impl <T: Float+TBlas>SoftmaxOps<T> for InternCPU {
    fn softmax(&self, inputs: &Matrix<T>) -> Matrix<T> {
        let exp = self.exp(&self.sub_col(&inputs, &self.max_cols(inputs)));
        self.div_col(&exp, &self.sum_cols(&exp))
    }

    #[cfg(not(feature="safe"))]
    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T> {
        let mut data = cached(self, grads.dims());

        let rows = grads.rows();
        let cols = grads.cols();
    
        let activated_data = activated.as_cpu_slice();
        let grad_data = grads.as_cpu_slice();

        let data_slice = data.as_cpu_slice_mut();

        for idx in range(rows-1) {
            let index = idx*cols;
            
            let single_out = Matrix::from(( (&activated_data[index..index+cols]).as_ptr() as *mut T, (cols, 1)));    
            let single_grad = Matrix::from(( (&grad_data[index..index+cols]).as_ptr() as *mut T, (cols, 1)));
        
            let diagflat = self.diagflat(&single_out);

            let jacobian_matrix = self.sub(&diagflat, &self.gemm(&single_out, &self.transpose(&single_out)));

            let res = self.gemm(&jacobian_matrix, &single_grad);

            let data_row = &mut data_slice[index..index+cols];
            data_row.copy_from_slice(res.as_cpu_slice());
        }
        data
    }
    #[cfg(feature="safe")]
    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T> {
        use custos::CPU;
        let device = CPU::new();
        let mut data = cached(self, grads.dims());
        
        let rows = grads.rows();
        let cols = grads.cols();
    
        let activated_data = activated.as_cpu_slice();
        let grad_data = grads.as_cpu_slice();

        let data_slice = data.as_cpu_slice_mut();

        for idx in range(rows-1) {
            let index = idx*cols;
            
            let single_out = Matrix::from((&device, (cols, 1), activated_data[index..index+cols].to_vec()));
            //let single_out = Matrix::from(( (&activated_data[index..index+cols]).as_ptr() as *mut T, (cols, 1)));    
            let single_grad = Matrix::from((&device, (cols, 1), grad_data[index..index+cols].to_vec()));
            //let single_grad = Matrix::from(( (&grad_data[index..index+cols]).as_ptr() as *mut T, (cols, 1)));
        
            let diagflat = self.diagflat(&single_out);

            let jacobian_matrix = self.sub(&diagflat, &self.gemm(&single_out, &self.transpose(&single_out)));

            let res = self.gemm(&jacobian_matrix, &single_grad);

            let data_row = &mut data_slice[index..index+cols];
            data_row.copy_from_slice(res.as_cpu_slice());
        }
        data
    }
}

impl <T: GenericOCL+TBlas+Float>SoftmaxOps<T> for InternCLDevice {
    fn softmax(&self, inputs: &Matrix<T>) -> Matrix<T> {
        switch_to_cpu_help_s(self, inputs, |device, inputs| device.softmax(&inputs))
    }

    fn softmax_grad(&self, activated: &Matrix<T>, grads: &Matrix<T>) -> Matrix<T> {
        switch_to_cpu_help_lr(self, activated, &grads, |device, activated, grads| device.softmax_grad(activated, &grads))
    }
}
