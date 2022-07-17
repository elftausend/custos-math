use custos::{get_device, libs::cpu::CPU, AsDev, Device, DeviceError, Error, VecRead};
#[cfg(feature = "opencl")]
use custos::{libs::opencl::CLDevice, BaseDevice};
use custos_math::{BaseOps, Matrix};

#[test]
fn test_matrix_read_cpu() -> Result<(), Error> {
    let device = CPU::new().select();

    let read = get_device!(VecRead<f32>)?;

    let matrix = Matrix::from((&device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(matrix.as_buf());
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_matrix_read_cl() -> Result<(), Error> {
    let device = CLDevice::new(0)?.select();

    let read = get_device!(VecRead<f32>)?;

    let matrix = Matrix::from((&device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(matrix.as_buf());
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);

    let base_device = get_device!(BaseDevice<f32>)?;
    assert_eq!(&read, &base_device.read(matrix.as_buf()));
    Ok(())
}

#[test]
fn test_no_device() {
    let device = get_device!(BaseOps<f32>);
    match &device {
        Ok(_) => panic!("Should not panic, as no device is selected"),
        Err(e) => assert!(e.kind() == Some(&DeviceError::NoDeviceSelected)),
    };

    {
        let device = CPU::new().select();

        let a = Matrix::from((&device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
        let b = Matrix::from((&device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));

        let c = device.add(&a, &b);
        assert_eq!(c.read(), vec![3.02, 12.246, 14., 10.42, 17.24, 9.53]);

        let device = get_device!(BaseOps<f32>);
        match device {
            Ok(_) => {}
            Err(_) => panic!("Should not panic, as a device is selected"),
        };
    }

    match device {
        Ok(_) => panic!("Should not panic, as no device is selected"),
        Err(_) => {}
    };
}

#[test]
fn test_baseops() -> Result<(), Error> {
    let device = CPU::new().select();

    let matrix = Matrix::from((&device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let b = Matrix::from((&device, (2, 3), [1., 1., 1., 1., 1., 1.]));

    let base_ops = get_device!(BaseOps<f32>)?;
    let out = base_ops.add(&matrix, &b);

    assert_eq!(out.read(), vec![2.51, 7.123, 8., 6.21, 9.62, 5.765]);
    Ok(())
}

#[test]
fn test_get_dev2() {
    let _device = CPU::new().select();
    let _dev = get_device!(Device<f32>).unwrap();
}
