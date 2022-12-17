use crate::{AdditionalOps, BaseOps, ClipOp, FnsOps, Matrix, SumOps};
use custos::{number::Float, CDatatype, CPU};

#[cfg(feature = "opencl")]
use custos::OpenCL;

pub trait CCE<T> {
    fn cce(&self, targets: &Matrix<T>) -> (T, Matrix<T>);
}


/* 

impl<T: Float + CDatatype> CCE<T> for Matrix<'_, T>
where
    Box<dyn CCEOp<T>>: CCEOp<T>,
{
    fn cce(&self, targets: &Matrix<T>) -> (T, Matrix<T>) {
        let device = get_device!(self.device(), CCEOp<T>);
        let loss = cce(device, self, targets);
        let grad = cce_grad(device, self, targets);
        (loss, grad)
    }
}

pub trait CCEOp<T>: FnsOps<T> + ClipOp<T> + BaseOps<T> + SumOps<T> + AdditionalOps<T> {}
impl<T: Float + CDatatype> CCEOp<T> for CPU {}
#[cfg(feature = "opencl")]
impl<T: Float + CDatatype> CCEOp<T> for OpenCL {}
#[cfg(feature = "cuda")]
impl<T: Float + CDatatype> CCEOp<T> for custos::CudaDevice {}

pub fn cce<T: Float>(device: &dyn CCEOp<T>, preds: &Matrix<T>, targets: &Matrix<T>) -> T {
    let preds = device.clip(preds, T::as_generic(1E-7), T::as_generic(1. - 1E-7));
    let confidences = device.sum_cols(&device.mul(&preds, targets));
    device.mean(&device.neg(&device.ln(&confidences)))
}

pub fn cce_grad<'a, T: Float>(
    device: &'a dyn CCEOp<T>,
    preds: &Matrix<T>,
    targets: &Matrix<T>,
) -> Matrix<'a, T> {
    let grad = device.neg(&device.div(targets, preds));
    device.divs(&grad, T::from_usize(preds.rows()))
}
*/