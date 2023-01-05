use crate::{AdditionalOps, BaseOps, ClipOp, FnsOps, Matrix, SumOps};
use custos::{number::Float, Device};

pub trait CCE<T> {
    fn cce(&self, targets: &Matrix<T>) -> (T, Matrix<T>);
}

impl<'a, T, D> Matrix<'a, T, D> where D: Device {}

pub trait CCEOp<T, D = Self>: Device
where
    D: Device,
{
    fn cce<'a>(
        &self,
        preds: &Matrix<'a, T, D>,
        targets: &Matrix<'a, T, D>,
    ) -> (T, Matrix<'a, T, Self>);
    fn cce_loss(&self, preds: &Matrix<T, D>, targets: &Matrix<T, D>) -> T;
    fn cce_grad<'a>(
        &self,
        preds: &Matrix<'a, T, D>,
        targets: &Matrix<'a, T, D>,
    ) -> Matrix<'a, T, Self>;
}

impl<'a, T, D: CCEOp<T>> Matrix<'a, T, D> {
    pub fn cce(&self, targets: &Matrix<'a, T, D>) -> (T, Matrix<'a, T, D>) {
        self.device().cce(self, targets)
    }

    pub fn cce_loss(&self, targets: &Matrix<T, D>) -> T {
        self.device().cce_loss(self, targets)
    }

    pub fn cce_grad(&self, targets: &Matrix<'a, T, D>) -> Matrix<'a, T, D> {
        self.device().cce_grad(self, targets)
    }
}

impl<T, D> CCEOp<T, D> for D
where
    T: Float,
    D: FnsOps<T> + ClipOp<T> + BaseOps<T> + SumOps<T> + AdditionalOps<T>,
{
    fn cce<'a>(
        &self,
        preds: &Matrix<'a, T, D>,
        targets: &Matrix<'a, T, D>,
    ) -> (T, Matrix<'a, T, Self>) {
        let loss = self.cce_loss(preds, targets);
        let grad = self.cce_grad(preds, targets);

        (loss, grad)
    }

    fn cce_loss(&self, preds: &Matrix<T, D>, targets: &Matrix<T, D>) -> T {
        let preds = preds.clip(T::as_generic(1E-7), T::as_generic(1. - 1E-7));
        let confidences = (&preds * targets).sum_cols();
        confidences.ln().neg().mean()
    }

    fn cce_grad<'a>(
        &self,
        preds: &Matrix<'a, T, D>,
        targets: &Matrix<'a, T, D>,
    ) -> Matrix<'a, T, Self> {
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
