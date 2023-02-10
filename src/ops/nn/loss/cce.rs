use crate::{AdditionalOps, BaseOps, ClipOp, FnsOps, Matrix, SumOps, SumOverOps};
use custos::{number::Float, Device, Shape};

pub trait CCE<T> {
    fn cce(&self, targets: &Matrix<T>) -> (T, Matrix<T>);
}

impl<'a, T, D> Matrix<'a, T, D> where D: Device {}

pub trait CCEOp<T, S: Shape = (), D = Self>: Device
where
    D: Device,
{
    #[inline]
    fn cce<'a>(
        &self,
        preds: &Matrix<'a, T, D, S>,
        targets: &Matrix<'a, T, D, S>,
    ) -> (T, Matrix<'a, T, Self, S>) {
        (self.cce_loss(preds, targets), self.cce_grad(preds, targets))
    }
    fn cce_loss(&self, preds: &Matrix<T, D, S>, targets: &Matrix<T, D, S>) -> T;
    fn cce_grad<'a>(
        &self,
        preds: &Matrix<'a, T, D, S>,
        targets: &Matrix<'a, T, D, S>,
    ) -> Matrix<'a, T, Self, S>;
}

impl<'a, T, S: Shape, D: CCEOp<T, S>> Matrix<'a, T, D, S> {
    #[inline]
    pub fn cce(&self, targets: &Matrix<'a, T, D, S>) -> (T, Matrix<'a, T, D, S>) {
        self.device().cce(self, targets)
    }

    #[inline]
    pub fn cce_loss(&self, targets: &Matrix<T, D, S>) -> T {
        self.device().cce_loss(self, targets)
    }

    #[inline]
    pub fn cce_grad(&self, targets: &Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S> {
        self.device().cce_grad(self, targets)
    }
}

impl<T, D, IS: Shape> CCEOp<T, IS> for D
where
    T: Float,
    D: FnsOps<T>
        + ClipOp<T, IS>
        + BaseOps<T, IS>
        + SumOps<T>
        + SumOverOps<T, IS>
        + AdditionalOps<T, IS>
        + FnsOps<T, IS>,
{
    fn cce_loss(&self, preds: &Matrix<T, D, IS>, targets: &Matrix<T, D, IS>) -> T {
        let preds = preds.clip(T::as_generic(1E-7), T::as_generic(1. - 1E-7));
        let confidences = (&preds * targets).sum_cols::<()>();
        confidences.ln().neg().mean()
    }

    fn cce_grad<'a>(
        &self,
        preds: &Matrix<'a, T, D, IS>,
        targets: &Matrix<'a, T, D, IS>,
    ) -> Matrix<'a, T, Self, IS> {
        let grad = (targets / preds).neg();
        grad / T::from_usize(preds.rows())
    }
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
impl<T: Float + CDatatype> CCEOp<T> for custos::CUDA {}

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
