#[cfg(not(feature = "realloc"))]
use custos::{devices::cpu::CPU, range, Read};

#[cfg(not(feature = "realloc"))]
#[cfg(feature = "opencl")]
use custos::devices::opencl::OpenCL;

#[cfg(not(feature = "realloc"))]
#[test]
fn test_threading_cpu() {
    let device = CPU::new();

    let th1_cl = std::thread::spawn(|| {
        let device = CPU::new();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(500) {
            let c = &a * &b;
            assert_eq!(device.read(&c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }

        assert_eq!(device.cache.borrow().nodes.len(), 1);

        for _ in range(500) {
            let c = &a - &b;
            let d = &a + &b + &c;
            let e = &a * &b - &c + &d * &d - &a;
            assert_eq!(34., e.read()[0]);
        }
        assert_eq!(device.cache.borrow().nodes.len(), 8);

        let c = &a - &b;
        let d = &a + &b + &c;
        let e = &a * &b - &c + &d * &d - &a;
        assert_eq!(34., e.read()[0]);
        assert_eq!(device.cache.borrow().nodes.len(), 8);
    });

    let th1_cpu = std::thread::spawn(|| {
        let device = CPU::new();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(500) {
            let c = &a * &b;
            assert_eq!(device.read(&c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        assert_eq!(device.cache.borrow().nodes.len(), 1);

        for _ in range(500) {
            let c = &a - &b;
            let d = &a + &b + &c;
            let e = &a * &b - &c + &d * &d - &a;
            assert_eq!(34., e.read()[0]);
        }
        assert_eq!(device.cache.borrow().nodes.len(), 8);

        let c = &a - &b;
        let d = &a + &b + &c;
        let e = &a * &b - &c + &d * &d - &a;
        assert_eq!(34., e.read()[0]);
        assert_eq!(device.cache.borrow().nodes.len(), 8);
    });

    let th2 = std::thread::spawn(|| {
        {
            let device = CPU::new();

            let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
            let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

            for _ in range(500) {
                let c = &a + &b;
                assert_eq!(device.read(&c.as_buf()), vec![4., 5., 3., 11., 11., 8.]);

                for _ in range(5) {
                    let d = &a * &b * &c;
                    let _ = &d + &c - (&b + &a * &d);
                }
                assert_eq!(device.cache.borrow().nodes.len(), 7);
            }
        } //'device' is dropped

        // assert_eq!(device.cache.borrow().nodes.len(), 0);
    });

    let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
    let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

    for _ in range(500) {
        let c = &a - &b;
        assert_eq!(c.read(), vec![2., -1., -1., -1., 1., 0.]);
    }

    assert_eq!(device.cache.borrow().nodes.len(), 1);

    use custos_math::Matrix;

    th1_cl.join().unwrap();
    th1_cpu.join().unwrap();
    th2.join().unwrap();
}

#[cfg(not(feature = "realloc"))]
#[cfg(feature = "opencl")]
#[test]
fn test_threading_cl_a() {
    use custos_math::Matrix;
    let device = OpenCL::new(0).unwrap();

    let th1_cl = std::thread::spawn(|| {
        let device = OpenCL::new(0).unwrap();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(100) {
            let c = &a * &b;
            assert_eq!(device.read(&c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        assert_eq!(device.cache.borrow().nodes.len(), 1);

        for _ in range(100) {
            let c = &a - &b;
            let d = &a + &b + &c;
            let e = &a * &b - &c + &d * &d - &a;
            assert_eq!(34., e.read()[0]);
        }
        assert_eq!(device.cache.borrow().nodes.len(), 8);

        let c = &a - &b;
        let d = &a + &b + &c;
        let e = &a * &b - &c + &d * &d - &a;
        assert_eq!(34., e.read()[0]);
        assert_eq!(device.cache.borrow().nodes.len(), 8);
    });

    let th1_cpu = std::thread::spawn(|| {
        let device = CPU::new();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(100) {
            let c = &a * &b;
            assert_eq!(device.read(&c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        assert_eq!(device.cache.borrow().nodes.len(), 1);

        for _ in range(100) {
            let c = &a - &b;
            let d = &a + &b + &c;
            let e = &a * &b - &c + &d * &d - &a;
            assert_eq!(34., e.read()[0]);
        }
        assert_eq!(device.cache.borrow().nodes.len(), 8);

        let c = &a - &b;
        let d = &a + &b + &c;
        let e = &a * &b - &c + &d * &d - &a;
        assert_eq!(34., e.read()[0]);
        assert_eq!(device.cache.borrow().nodes.len(), 8);
    });

    let th2 = std::thread::spawn(|| {
        {
            let device = CPU::new();

            let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
            let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

            for _ in range(100) {
                let c = &a + &b;
                assert_eq!(device.read(&c.as_buf()), vec![4., 5., 3., 11., 11., 8.]);

                for _ in range(5) {
                    let d = &a * &b * &c;
                    let _ = &d + &c - (&b + &a * &d);
                }
                assert_eq!(device.cache.borrow().nodes.len(), 7);
            }
        } //'device' is dropped
    });

    let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
    let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

    for _ in range(100) {
        let c = &a - &b;
        assert_eq!(c.read(), vec![2., -1., -1., -1., 1., 0.]);
    }
    // CL_CACHE.with(|f| assert!(f.borrow().output_nodes.len() == 1));
    th1_cl.join().unwrap();
    th1_cpu.join().unwrap();
    th2.join().unwrap();
}

#[cfg(not(feature = "realloc"))]
#[cfg(feature = "cuda")]
#[test]
fn test_threading_cuda_a() -> custos::Result<()> {
    use custos::CUDA;
    use custos_math::Matrix;
    use std::thread::JoinHandle;

    let device = CUDA::new(0)?;

    let th1: JoinHandle<Result<(), custos::Error>> = std::thread::spawn(|| {
        let device = CUDA::new(0)?;

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(100) {
            let c = &a * &b;
            assert_eq!(device.read(&c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        assert_eq!(device.cache.borrow().nodes.len(), 1);

        for _ in range(100) {
            let c = &a - &b;
            let d = &a + &b + &c;
            let e = &a * &b - &c + &d * &d - &a;
            assert_eq!(34., e.read()[0]);
        }
        assert_eq!(device.cache.borrow().nodes.len(), 8);

        let c = &a - &b;
        let d = &a + &b + &c;
        let e = &a * &b - &c + &d * &d - &a;
        assert_eq!(34., e.read()[0]);
        assert_eq!(device.cache.borrow().nodes.len(), 8);
        Ok(())
    });

    let th1_cuda = std::thread::spawn(|| {
        let device = CUDA::new(0).unwrap();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        for _ in range(100) {
            let c = &a * &b;
            assert_eq!(device.read(&c.as_buf()), vec![3., 6., 2., 30., 30., 16.]);
        }
        assert_eq!(device.cache.borrow().nodes.len(), 1);
    });

    let th2 = std::thread::spawn(|| {
        let device = CPU::new();

        let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
        let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

        let c = &a + &b;
        assert_eq!(device.read(&c.as_buf()), vec![4., 5., 3., 11., 11., 8.]);

        let d = &a * &b * &c;
        let _ = &d + &c - (&b + &a * &d);

        assert_eq!(device.cache.borrow().nodes.len(), 7);
    });

    let a = Matrix::from((&device, (3, 2), [3f32, 2., 1., 5., 6., 4.]));
    let b = Matrix::from((&device, (2, 3), [1., 3., 2., 6., 5., 4.]));

    let c = &a - &b;
    assert_eq!(c.read(), vec![2., -1., -1., -1., 1., 0.]);

    th1.join().unwrap()?;
    th1_cuda.join().unwrap();
    th2.join().unwrap();
    Ok(())
}
