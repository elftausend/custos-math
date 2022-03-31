use custos::{Matrix, InternCPU, number::{Number, Float}, range, Gemm, cpu::TBlas, BaseOps, opencl::GenericOCL, InternCLDevice};
use crate::{SumOps, MaxOps, ColOp, FnsOps, DiagflatOp, TransposeOp, ops::{switch_to_cpu_help_s, switch_to_cpu_help_lr}};

pub trait Softmax<T> {
    fn softmax(&self, inputs: Matrix<T>) -> Matrix<T>;
    fn softmax_grad(&self, activated: Matrix<T>, grads: Matrix<T>) -> Matrix<T>;
}

impl <T: Float+TBlas>Softmax<T> for InternCPU {
    fn softmax(&self, inputs: Matrix<T>) -> Matrix<T> {
        let exp = self.sub_col(inputs, self.exp(self.max_cols(inputs)));
        self.div_col(exp, self.sum_cols(exp))
    }

    fn softmax_grad(&self, mut activated: Matrix<T>, mut grads: Matrix<T>) -> Matrix<T> {
        let mut data = Matrix::new(self.clone(), grads.dims());

        let rows = grads.rows();
        let cols = grads.cols();

        let activated_data = activated.as_cpu_slice_mut();
        let grad_data = grads.as_cpu_slice_mut();

        let data_slice = data.as_cpu_slice_mut();

        for idx in range(rows) {
            let index = idx*cols;
            
            let single_out = Matrix::from(( (&mut activated_data[index..index+cols]).as_mut_ptr(), (1, cols)));    
            let single_gard = Matrix::from(( (&mut grad_data[index..index+cols]).as_mut_ptr(), (1, cols)));
        
            let diagflat = self.diagflat(single_out);

            let jacobian_matrix = self.sub(diagflat, self.gemm(single_out, self.transpose(single_out)));

            let res = self.gemm(jacobian_matrix, single_gard);

            let data_row = &mut data_slice[index..index+cols];
            data_row.copy_from_slice(res.as_cpu_slice());
        }
        data
    }
}

impl <T: GenericOCL+TBlas+Float>Softmax<T> for InternCLDevice {
    fn softmax(&self, inputs: Matrix<T>) -> Matrix<T> {
        switch_to_cpu_help_s(self, inputs, |device, inputs| device.softmax(inputs))
    }

    fn softmax_grad(&self, activated: Matrix<T>, grads: Matrix<T>) -> Matrix<T> {
        switch_to_cpu_help_lr(self, activated, grads, |device, activated, grads| device.softmax_grad(activated, grads))
    }
}
