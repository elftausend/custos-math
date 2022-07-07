use crate::{AdditionalOps, ClipOp, FnsOps, SumOps};
use custos::{get_device, number::Float, BaseOps, CDatatype, CLDevice, CPU, Matrix};

pub trait CCE<T> {
    fn cce(&self, targets: &Matrix<T>) -> (T, Matrix<T>);
}

impl<T: Float + CDatatype> CCE<T> for Matrix<T>
where
    Box<dyn CCEOp<T>>: CCEOp<T>,
{
    fn cce(&self, targets: &Matrix<T>) -> (T, Matrix<T>) {
        let device = get_device!(CCEOp, T).unwrap();
        let loss = cce(&device, self, targets);
        let grad = cce_grad(&device, self, targets);
        (loss, grad)
    }
}

pub trait CCEOp<T>: FnsOps<T> + ClipOp<T> + BaseOps<T> + SumOps<T> + AdditionalOps<T> {}
impl<T: Float + CDatatype> CCEOp<T> for CPU {}
#[cfg(feature="opencl")]
impl<T: Float + CDatatype> CCEOp<T> for CLDevice {}
#[cfg(feature="cuda")]
impl<T: Float + CDatatype> CCEOp<T> for custos::CudaDevice {}

pub fn cce<T: Float>(device: &dyn CCEOp<T>, preds: &Matrix<T>, targets: &Matrix<T>) -> T {
    let preds = device.clip(preds, T::as_generic(1E-7), T::as_generic(1. - 1E-7));
    let confidences = device.sum_cols(&device.mul(&preds, targets));
    device.mean(&device.neg(&device.ln(&confidences)))
}

pub fn cce_grad<T: Float>(
    device: &dyn CCEOp<T>,
    preds: &Matrix<T>,
    targets: &Matrix<T>,
) -> Matrix<T> {
    let grad = device.neg(&device.div(targets, &preds));
    device.divs(&grad, T::from_usize(preds.rows()))
}
