#[cfg(not(feature = "realloc"))]
#[cfg(feature = "opencl")]
#[test]
fn test_use_range_for_ew_add() {
    use custos::{get_count, range, OpenCL, Read};
    use custos_math::Matrix;
    let device = OpenCL::new(0).unwrap();

    let a = Matrix::from((&device, (1, 4), [1i32, 4, 2, 9]));
    let b = Matrix::from((&device, (1, 4), [1, 4, 2, 9]));

    let z = Matrix::from((&device, (1, 4), [1, 2, 3, 4]));

    for _ in range(100) {
        let c = &a + &b;
        assert_eq!(vec![2, 8, 4, 18], device.read(c.as_buf()));
        let d = &c + &z;
        assert_eq!(vec![3, 10, 7, 22], device.read(d.as_buf()));

        assert!(get_count() == 2);
    }

    assert!(get_count() == 0);

    let a = Matrix::from((&device, (1, 5), [1, 4, 2, 9, 1]));
    let b = Matrix::from((&device, (1, 5), [1, 4, 2, 9, 1]));

    let z = Matrix::from((&device, (1, 5), [1, 2, 3, 4, 5]));

    for _ in range(100) {
        let c = &a + &b;
        assert_eq!(vec![2, 8, 4, 18, 2], device.read(c.as_buf()));
        let d = &c + &z;
        assert_eq!(vec![3, 10, 7, 22, 7], device.read(d.as_buf()));

        assert!(get_count() == 2);
    }
    assert!(get_count() == 0);
}

#[cfg(feature = "safe")]
#[cfg(feature = "opencl")]
#[test]
fn test_use_range_for_ew_add() {
    use custos::{range, AsDev, OpenCL, VecRead};
    use custos_math::Matrix;

    let device = OpenCL::new(0).unwrap();

    let a = Matrix::from((&device, (1, 4), [1i32, 4, 2, 9]));
    let b = Matrix::from((&device, (1, 4), [1, 4, 2, 9]));

    let z = Matrix::from((&device, (1, 4), [1, 2, 3, 4]));

    for _ in range(100) {
        let c = &a + &b;
        assert_eq!(vec![2, 8, 4, 18], device.read(c.as_buf()));
        let d = &c + &z;
        assert_eq!(vec![3, 10, 7, 22], device.read(d.as_buf()));
    }

    let a = Matrix::from((&device, (1, 5), [1, 4, 2, 9, 1]));
    let b = Matrix::from((&device, (1, 5), [1, 4, 2, 9, 1]));

    let z = Matrix::from((&device, (1, 5), [1, 2, 3, 4, 5]));

    for _ in range(100) {
        let c = &a + &b;
        assert_eq!(vec![2, 8, 4, 18, 2], device.read(c.as_buf()));
        let d = &c + &z;
        assert_eq!(vec![3, 10, 7, 22, 7], device.read(d.as_buf()));
    }
}

#[cfg(not(feature = "realloc"))]
#[test]
fn test_nested_for() {
    use custos::{get_count, range, CPU};
    use custos_math::Matrix;
    let device = CPU::new();

    let a = Matrix::from((&device, (1, 5), [1i32, 4, 2, 9, 1]));
    let b = Matrix::from((&device, (1, 5), [1, 4, 2, 9, 1]));

    for _ in range(100) {
        let c = &a + &b;
        for _ in range(200) {
            let d = &c + &b;
            let e = &a + &b + &c + &d;
            assert!(get_count() == 5);

            for _ in range(10) {
                let _ = &d + &e;
                assert!(get_count() == 6);
            }
        }
        assert!(get_count() == 1)
    }

    assert!(get_count() == 0);
}

#[cfg(feature = "safe")]
#[test]
fn test_nested_for() {
    use custos::{range, AsDev, CPU};
    use custos_math::Matrix;

    let device = CPU::new();

    let a = Matrix::from((&device, (1, 5), [1i32, 4, 2, 9, 1]));
    let b = Matrix::from((&device, (1, 5), [1, 4, 2, 9, 1]));

    for _ in range(100) {
        let c = &a + &b;
        for _ in range(200) {
            let d = &c + &b;
            let e = &a + &b + (&c + &d);

            for _ in range(10) {
                let _ = &d + &e;
            }
        }
    }
}
