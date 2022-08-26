use custos::{number::Number, CDatatype};

use crate::{AdditionalOps, BaseOps, Matrix, SumOps};

pub fn mse<T: CDatatype, D: BaseOps<T> + SumOps<T>>(
    preds: &Matrix<T>,
    targets: &Matrix<T>,
) -> T {
    let x = preds-targets;
    (&x * &x).mean()
}

pub fn mse_grad<T: CDatatype, D: BaseOps<T> + AdditionalOps<T>>(
    preds: &Matrix<T>,
    targets: &Matrix<T>,
) -> Matrix<T> {
    let x = preds - targets;
    (&x * T::two() / T::from_usize(preds.cols())) / T::from_usize(preds.rows())
    
}


