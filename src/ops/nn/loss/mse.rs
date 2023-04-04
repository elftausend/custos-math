use crate::{AdditionalOps, BaseOps, Matrix, SumOps};
#[cfg(feature = "opencl")]
use custos::{opencl::enqueue_kernel, OpenCL};
use custos::{prelude::Number, CDatatype, IsShapeIndep, Shape};

#[inline]
pub fn mse<'a, T, D, S>(
    preds: &Matrix<'a, T, D, S>,
    targets: &Matrix<'a, T, D>,
) -> (T, Matrix<'a, T, D>)
where
    T: Number,
    D: IsShapeIndep + BaseOps<T> + SumOps<T> + AdditionalOps<T>,
    S: Shape,
{
    let preds = preds.as_dims();
    (mse_loss(preds, targets), mse_grad(preds, targets))
}

pub fn mse_loss<T, D, S>(preds: &Matrix<T, D, S>, targets: &Matrix<T, D, S>) -> T
where
    D: BaseOps<T, S> + SumOps<T, S>,
    S: Shape,
{
    let x = preds - targets;
    (&x * &x).mean()
}

pub fn mse_grad<'a, T, D, S>(
    preds: &Matrix<'a, T, D, S>,
    targets: &Matrix<'a, T, D, S>,
) -> Matrix<'a, T, D, S>
where
    T: Number,
    D: BaseOps<T, S> + SumOps<T, S> + AdditionalOps<T, S>,
    S: Shape,
{
    let x = preds - targets;
    (&x * T::two() / T::from_usize(preds.cols())) / T::from_usize(preds.rows())
}

#[cfg(feature = "opencl")]
pub fn mse_grad_cl<'a, T: CDatatype + Number>(
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
        device.retrieve(preds.len(), (preds.as_buf(), targets.as_buf()));

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
