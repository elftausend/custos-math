use custos::number::Number;

use crate::{AdditionalOps, BaseOps, Matrix, SumOps};

pub fn mse<T: Copy, D: BaseOps<T> + SumOps<T>>(
    device: &D,
    preds: &Matrix<T>,
    targets: &Matrix<T>,
) -> T {
    let x = device.sub(preds, targets);
    device.mean(&device.mul(&x, &x))
}

pub fn mse_grad<T: Number, D: BaseOps<T> + AdditionalOps<T>>(
    device: &D,
    preds: &Matrix<T>,
    targets: &Matrix<T>,
) -> Matrix<T> {
    let x = device.sub(preds, targets);
    device.divs(
        &device.divs(&device.muls(&x, T::two()), T::from_usize(preds.cols())),
        T::from_usize(preds.rows()),
    )
}
