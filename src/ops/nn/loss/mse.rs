
#[cfg(feature="opencl")]
use custos::{opencl::enqueue_kernel, CLDevice, Cache};
use custos::CDatatype;
use crate::Matrix;

pub fn mse<T: CDatatype>(preds: &Matrix<T>, targets: &Matrix<T>) -> T {
    let x = preds - targets;
    (&x * &x).mean()
}

pub fn mse_grad<'a, T: CDatatype>(preds: &Matrix<'a, T>, targets: &Matrix<'a, T>) -> Matrix<'a, T> {
    let x = preds - targets;
    (&x * T::two() / T::from_usize(preds.cols())) / T::from_usize(preds.rows())
}

#[cfg(feature="opencl")]
pub fn mse_grad_cl<'a, T: CDatatype>(
    device: &'a CLDevice,
    preds: &Matrix<'a, T>,
    targets: &Matrix<'a, T>,
) -> Matrix<'a, T> {
    let src = format!(
        "
        __kernel void mse_grad(__global const {datatype}* preds, 
            __global const {datatype}* targets, 
            __global {datatype}* out,
            const {datatype} cols, const {datatype} rows) 
            
        {{
            size_t id = get_global_id(0);

            {datatype} x = (preds[id] - targets[id]) * 2;
            out[id] = (x / cols) / rows;
        }}
    ",
        datatype = T::as_c_type_str()
    );

    let out = Cache::get::<T, _>(device, preds.len, (preds.node.idx, targets.node.idx));
    enqueue_kernel(
        device,
        &src,
        [preds.len, 0, 0],
        None,
        &[
            preds,
            targets,
            &out,
            &T::from_usize(preds.cols()),
            &T::from_usize(preds.rows()),
        ],
    )
    .unwrap();
    (out, preds.dims()).into()
}
