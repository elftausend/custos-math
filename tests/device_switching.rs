#[cfg(feature = "opencl")]
use std::ffi::c_void;

#[cfg(feature = "opencl")]
use custos::opencl::api::{enqueue_write_buffer, wait_for_event};
#[cfg(feature = "opencl")]
use custos::{cpu::CPU, opencl::OpenCL, range};

#[cfg(feature = "opencl")]
#[test]
fn test_device_switching() -> Result<(), custos::Error> {
    use custos_math::{BaseOps, Matrix};

    let device = OpenCL::new(0)?;
    let a = Matrix::from((&device, (2, 3), [1.51f32, 6.123, 7., 5.21, 8.62, 4.765]));
    let b = Matrix::from((&device, (2, 3), [1.51f32, 6.123, 7., 5.21, 8.62, 4.765]));

    for _ in range(500) {
        let c = &a + &b;

        let cpu = CPU::new();
        let c = Matrix::from((&cpu, c.dims(), c.read()));
        let d_cpu = cpu.add(&c, &c);

        let d = Matrix::from((&device, d_cpu));
        assert_eq!(vec![6.04, 24.492, 28., 20.84, 34.48, 19.06], d.read());
    }
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_device_switching_s() -> Result<(), custos::Error> {
    use custos::cache::Cache;
    use custos_math::{BaseOps, Matrix};

    let device = OpenCL::new(0)?;
    let a = Matrix::from((&device, (2, 3), [1.51f32, 6.123, 7., 5.21, 8.62, 4.765]));
    let b = Matrix::from((&device, (2, 3), [1.51f32, 6.123, 7., 5.21, 8.62, 4.765]));

    let c = a + b;

    let cpu = CPU::new();
    let c = Matrix::from((&cpu, c.dims(), c.read()));
    let d_cpu = cpu.add(&c, &c);

    let out = Cache::get::<f32, ()>(&device, d_cpu.size(), c.node.idx);
    let event = unsafe {
        enqueue_write_buffer(
            &device.queue(),
            out.ptr.ptr as *mut c_void,
            d_cpu.as_slice(),
            true,
        )?
    };
    wait_for_event(event)?;
    let m = Matrix::from((out, d_cpu.dims()));
    assert_eq!(vec![6.04, 24.492, 28., 20.84, 34.48, 19.06], m.read());
    Ok(())
}
