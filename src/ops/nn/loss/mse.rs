use custos::CDatatype;

use crate::Matrix;

pub fn mse<T: CDatatype>(
    preds: &Matrix<T>,
    targets: &Matrix<T>,
) -> T {
    let x = preds-targets;
    (&x * &x).mean()
}

pub fn mse_grad<'a, T: CDatatype>(
    preds: &Matrix<'a, T>,
    targets: &Matrix<'a, T>,
) -> Matrix<'a, T> {
    let x = preds - targets;
    (&x * T::two() / T::from_usize(preds.cols())) / T::from_usize(preds.rows())
    
}


