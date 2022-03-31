use custos::{Matrix, number::Float, BaseOps};
use crate::{FnsOps, ClipOp, SumOps, AdditionalOps};

pub fn cce<T: Float, D: FnsOps<T>+ClipOp<T>+BaseOps<T>+SumOps<T>>(device: D, preds: Matrix<T>, targets: Matrix<T>) -> T {
    let preds = device.clip(preds, T::as_generic(1E-7), T::as_generic(1.-1E-7));
    let confidences = device.sum_cols(device.mul(preds, targets));
    device.mean(device.neg(device.ln(confidences)))
}

pub fn cce_grad<T: Float, D: FnsOps<T>+BaseOps<T>+AdditionalOps<T>>(device: D, preds: Matrix<T>, targets: Matrix<T>) -> Matrix<T> {
    let grad = device.neg(device.div(targets, preds));
    device.divs(grad, T::from_usize(preds.rows()))
}