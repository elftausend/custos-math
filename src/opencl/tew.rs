use custos::{
    opencl::enqueue_kernel,
    Buffer, CDatatype, CLDevice, cache::Cache,
};

trait Both {
    fn as_str<'a>() -> &'a str;
}

/*
impl <T: GenericOCL>Both for T {
    fn as_str<'a>() -> &'a str {
        T::as_ocl_type_str()
    }
}


impl <T: !GenericOCL>Both for T {
    fn as_str<'a, >() -> &'a str {
        "undefined"
    }
}
*/

//std::any::TypeId::of::<T>() ... check all impl

/// Element-wise operations. The op/operation is usually "+", "-", "*", "/".
/// "tensor element-wise"
///
/// # Example
/// ```
/// use custos::{CLDevice, Buffer, VecRead};
/// use custos_math::cl_tew;
///
/// fn main() -> Result<(), custos::Error> {
///     let device = CLDevice::new(0)?;
///     let lhs = Buffer::<i16>::from((&device, [15, 30, 21, 5, 8]));
///     let rhs = Buffer::<i16>::from((&device, [10, 9, 8, 6, 3]));
///
///     let result = cl_tew(&device, &lhs, &rhs, "+")?;
///     assert_eq!(vec![25, 39, 29, 11, 11], device.read(&result));
///     Ok(())
/// }
/// ```
pub fn cl_tew<'a, T: CDatatype>(
    device: &'a CLDevice,
    lhs: &Buffer<T>,
    rhs: &Buffer<T>,
    op: &str,
) -> custos::Result<Buffer<'a, T>> {
    let src = format!("
        __kernel void eop(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]{op}rhs[id];
        }}
    ", datatype=T::as_c_type_str());

    let gws = [lhs.len, 0, 0];
    let out = Cache::get::<T, _, _>(device, lhs.len, (lhs.node.idx, rhs.node.idx));
    enqueue_kernel(device, &src, gws, None, &[lhs, rhs, &out])?;
    Ok(out)
}

/// Element-wise "assign" operations. The op/operation is usually "+", "-", "*", "/".
///
/// # Example
/// ```
/// use custos::{CLDevice, Buffer, VecRead};
/// use custos_math::cl_tew_self;
///
/// fn main() -> Result<(), custos::Error> {
///     let device = CLDevice::new(0)?;
///     let mut lhs = Buffer::<i16>::from((&device, [15, 30, 21, 5, 8]));
///     let rhs = Buffer::<i16>::from((&device, [10, 9, 8, 6, 3]));
///
///     cl_tew_self(&device, &mut lhs, &rhs, "+")?;
///     assert_eq!(vec![25, 39, 29, 11, 11], device.read(&lhs));
///     Ok(())
/// }
/// ```
pub fn cl_tew_self<T: CDatatype>(
    device: &CLDevice,
    lhs: &mut Buffer<T>,
    rhs: &Buffer<T>,
    op: &str,
) -> custos::Result<()> {
    let src = format!(
        "
        __kernel void eop_self(__global {datatype}* self, __global const {datatype}* rhs) {{
            size_t id = get_global_id(0);
            self[id] = self[id]{op}rhs[id];
        }}
    ",
        datatype = T::as_c_type_str()
    );

    let gws = [lhs.len, 0, 0];
    enqueue_kernel(device, &src, gws, None, &[lhs, rhs])?;
    Ok(())
}
