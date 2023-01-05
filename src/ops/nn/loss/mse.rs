use crate::{AdditionalOps, BaseOps, Matrix, SumOps};
#[cfg(feature = "opencl")]
use custos::{opencl::enqueue_kernel, OpenCL};
use custos::{prelude::Number, CDatatype, Device};

pub fn mse<T, D: BaseOps<T> + SumOps<T>>(preds: &Matrix<T, D>, targets: &Matrix<T, D>) -> T {
    let x = preds - targets;
    (&x * &x).mean()
}

pub fn mse_grad<'a, T: Number, D: BaseOps<T> + SumOps<T> + AdditionalOps<T>>(
    preds: &Matrix<'a, T, D>,
    targets: &Matrix<'a, T, D>,
) -> Matrix<'a, T, D> {
    let x = preds - targets;
    (&x * T::two() / T::from_usize(preds.cols())) / T::from_usize(preds.rows())
}

#[cfg(feature = "opencl")]
pub fn mse_grad_cl<'a, T: CDatatype>(
    device: &'a OpenCL,
    preds: &Matrix<'a, T, OpenCL>,
    targets: &Matrix<'a, T, OpenCL>,
) -> Matrix<'a, T, OpenCL> {
    use custos::Device;

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

    let out: custos::Buffer<T, OpenCL> =
        device.retrieve(preds.len(), (preds.node.idx, targets.node.idx));
    enqueue_kernel(
        device,
        &src,
        [preds.len(), 0, 0],
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
