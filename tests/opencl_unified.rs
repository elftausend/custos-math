#[cfg(feature = "opencl")]
use std::ffi::c_void;

#[cfg(feature = "opencl")]
use custos::{
    opencl::api::{clCreateBuffer, MemFlags, OCLErrorKind},
    CLDevice, Error,
};

#[cfg(feature = "opencl")]
pub fn unified_mem<T>(device: &CLDevice, arr: &mut [T]) -> Result<*mut c_void, Error> {
    let mut err = 0;

    let r = unsafe {
        clCreateBuffer(
            device.ctx().0,
            MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr,
            arr.len() * core::mem::size_of::<T>(),
            arr.as_mut_ptr() as *mut c_void,
            &mut err,
        )
    };

    device.inner.borrow_mut().ptrs.push(r);

    if err != 0 {
        return Err(Error::from(OCLErrorKind::from_value(err)));
    }
    Ok(r)
}

#[cfg(feature = "opencl")]
#[test]
fn test_unified_mem_device_switch() -> custos::Result<()> {
    use custos::CLDevice;
    use custos_math::{cpu_exec, FnsOps, Matrix};

    let device = CLDevice::new(0)?;

    let a = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.]));
    let m = cpu_exec(&device, &a, |cpu, m| cpu.ln(m))?;
    println!("m: {m:?}");

    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_unified_opencl() -> custos::Result<()> {
    use custos::CLDevice;
    use custos_math::Matrix;

    let device = CLDevice::new(0)?;

    if !device.unified_mem() {
        return Ok(());
    }

    let mut a = Matrix::from((&device, 2, 3, [1, 2, 3, 4, 5, 6]));

    for (i, value) in a.as_mut_buf().iter_mut().enumerate() {
        *value += i as i32;
    }

    //let slice = unsafe { std::slice::from_raw_parts(a.as_buf().ptr.0, a.size()) };
    //println!("slice: {slice:?}");

    assert_eq!(a.read(), vec![1, 3, 5, 7, 9, 11]);
    //println!("a: {a:?}");
    Ok(())
}

#[cfg(not(feature = "safe"))]
#[cfg(feature = "opencl")]
#[test]
fn test_unified_calc() -> custos::Result<()> {
    use std::{marker::PhantomData, ptr::null_mut};

    use custos::{AsDev, BufFlag, Buffer, CLDevice, CPU, Node};
    use custos_math::cl_tew;

    let len = 100;

    let device = CPU::new();
    let mut a = Buffer::<f32>::new(&device, len);
    let mut b = Buffer::<f32>::from((&device, vec![1.; len]));

    let cl = CLDevice::new(0)?;

    let a: Buffer<f32> = Buffer {
        ptr: (null_mut(), unified_mem(&cl, a.as_mut_slice())?, 0),
        len,
        device: device.dev(),
        flag: BufFlag::Wrapper,
        node: Node::default(),
        p: PhantomData,
    };
    let b = Buffer {
        ptr: (null_mut(), unified_mem(&cl, b.as_mut_slice())?, 0),
        len,
        device: device.dev(),
        flag: BufFlag::Wrapper,
        node: Node::default(),
        p: PhantomData,
    };

    cl_tew(&cl, &a, &b, "+")?;

    //    let ptr = unified_ptr(cl.queue(), a)?;
    //    let ptr = unified_ptr(cl.queue(), a)?;

    Ok(())
}
