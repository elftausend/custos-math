#[cfg(not(feature = "safe"))]
use custos::{libs::cpu::CPU, range, AsDev, VecRead};

#[cfg(not(feature = "safe"))]
#[cfg(feature = "opencl")]
use custos::libs::opencl::CLDevice;

#[cfg(not(feature = "safe"))]
#[test]
fn test_threading_cpu() {
    let device = CPU::new().select();

    let th1_cl = std::thread::spawn(|| {
        let device = CPU::new().select();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(500) {
            let c = a * b;
            assert_eq!(device.read(c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 1));

        for _ in range(500) {
            let c = a - b;
            let d = a + b + c;
            let e = a * b - c + d * d - a;
            assert_eq!(34., e.read()[0]);
        }
        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 8));

        let c = a - b;
        let d = a + b + c;
        let e = a * b - c + d * d - a;
        assert_eq!(34., e.read()[0]);
        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 8));
    });

    let th1_cpu = std::thread::spawn(|| {
        let device = CPU::new().select();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(500) {
            let c = a * b;
            assert_eq!(device.read(c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 1));

        for _ in range(500) {
            let c = a - b;
            let d = a + b + c;
            let e = a * b - c + d * d - a;
            assert_eq!(34., e.read()[0]);
        }
        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 8));

        let c = a - b;
        let d = a + b + c;
        let e = a * b - c + d * d - a;
        assert_eq!(34., e.read()[0]);
        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 8));
    });

    let th2 = std::thread::spawn(|| {
        {
            let device = CPU::new().select();

            let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
            let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

            for _ in range(500) {
                let c = a + b;
                assert_eq!(device.read(c.as_buf()), vec![4., 5., 3., 11., 11., 8.]);

                for _ in range(5) {
                    let d = a * b * c;
                    let _ = d + c - (b + a * d);
                }
                CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 7));
            }
        } //'device' is dropped

        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 0));
    });

    let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
    let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

    for _ in range(500) {
        let c = a - b;
        assert_eq!(c.read(), vec![2., -1., -1., -1., 1., 0.]);
    }

    CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 1));

    use custos::cpu::CPU_CACHE;
    use custos_math::Matrix;

    th1_cl.join().unwrap();
    th1_cpu.join().unwrap();
    th2.join().unwrap();
}

#[cfg(not(feature = "safe"))]
#[cfg(feature = "opencl")]
#[test]
fn test_threading_cl_a() {
    use custos::{cpu::CPU_CACHE, opencl::CL_CACHE};
    use custos_math::Matrix;
    let device = CLDevice::new(0).unwrap().select();

    let th1_cl = std::thread::spawn(|| {
        let device = CLDevice::new(0).unwrap().select();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(100) {
            let c = a * b;
            assert_eq!(device.read(c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        CL_CACHE.with(|f| assert!(f.borrow().nodes.len() == 1));

        for _ in range(100) {
            let c = a - b;
            let d = a + b + c;
            let e = a * b - c + d * d - a;
            assert_eq!(34., e.read()[0]);
        }
        CL_CACHE.with(|f| assert!(f.borrow().nodes.len() == 8));

        let c = a - b;
        let d = a + b + c;
        let e = a * b - c + d * d - a;
        assert_eq!(34., e.read()[0]);
        CL_CACHE.with(|f| assert!(f.borrow().nodes.len() == 8));
    });

    let th1_cpu = std::thread::spawn(|| {
        let device = CPU::new().select();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(100) {
            let c = a * b;
            assert_eq!(device.read(c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 1));

        for _ in range(100) {
            let c = a - b;
            let d = a + b + c;
            let e = a * b - c + d * d - a;
            assert_eq!(34., e.read()[0]);
        }
        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 8));

        let c = &a - &b;
        let d = &a + &b + &c;
        let e = &a * &b - &c + &d * &d - &a;
        assert_eq!(34., e.read()[0]);
        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 8));
    });

    let th2 = std::thread::spawn(|| {
        {
            let device = CPU::new().select();

            let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
            let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

            for _ in range(100) {
                let c = a + b;
                assert_eq!(device.read(c.as_buf()), vec![4., 5., 3., 11., 11., 8.]);

                for _ in range(5) {
                    let d = a * b * c;
                    let _ = d + c - (b + a * d);
                }
                CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 7));
            }
        } //'device' is dropped

        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 0));
    });

    let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
    let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

    for _ in range(100) {
        let c = a - b;
        assert_eq!(c.read(), vec![2., -1., -1., -1., 1., 0.]);
    }
    // CL_CACHE.with(|f| assert!(f.borrow().output_nodes.len() == 1));
    th1_cl.join().unwrap();
    th1_cpu.join().unwrap();
    th2.join().unwrap();
}

#[cfg(not(feature = "safe"))]
#[cfg(feature = "cuda")]
#[test]
fn test_threading_cuda_a() -> custos::Result<()> {
    use custos::{cpu::CPU_CACHE, cuda::CUDA_CACHE, CudaDevice};
    use custos_math::Matrix;
    use std::thread::JoinHandle;

    let device = CudaDevice::new(0)?.select();

    let th1: JoinHandle<Result<(), custos::Error>> = std::thread::spawn(|| {
        let device = CudaDevice::new(0)?.select();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(100) {
            let c = a * b;
            assert_eq!(device.read(c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        CUDA_CACHE.with(|f| assert!(f.borrow().nodes.len() == 1));

        for _ in range(100) {
            let c = a - b;
            let d = a + b + c;
            let e = a * b - c + d * d - a;
            assert_eq!(34., e.read()[0]);
        }
        CUDA_CACHE.with(|f| assert!(f.borrow().nodes.len() == 8));

        let c = a - b;
        let d = a + b + c;
        let e = a * b - c + d * d - a;
        assert_eq!(34., e.read()[0]);
        CUDA_CACHE.with(|f| assert!(f.borrow().nodes.len() == 8));
        Ok(())
    });

    let th1_cuda = std::thread::spawn(|| {
        let device = CudaDevice::new(0).unwrap().select();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(100) {
            let c = a * b;
            assert_eq!(device.read(c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        CUDA_CACHE.with(|f| assert!(f.borrow().nodes.len() == 1));
    });

    let th2 = std::thread::spawn(|| {
        let device = CPU::new().select();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        let c = a + b;
        assert_eq!(device.read(c.as_buf()), vec![4., 5., 3., 11., 11., 8.]);

        let d = a * b * c;
        let _ = d + c - (b + a * d);

        CPU_CACHE.with(|f| assert!(f.borrow().nodes.len() == 7));
    });

    let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
    let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

    let c = a - b;
    assert_eq!(c.read(), vec![2., -1., -1., -1., 1., 0.]);

    th1.join().unwrap()?;
    th1_cuda.join().unwrap();
    th2.join().unwrap();
    Ok(())
}
