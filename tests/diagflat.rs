use custos::{cpu::CPU, opencl::CLDevice, AsDev, Matrix};
use custos_math::{DiagflatOp, cl_diagflat};

#[test]
fn test_diagflat_cpu() {
    let device = CPU::new().select();

    let x = Matrix::from((&device, (1, 4), [1.5, 2., 6., 3.]));
    let result = device.diagflat(&x);
    assert_eq!(
        result.read(),
        vec![1.5, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    );

    println!("res: {:?}", result);
}

#[test]
fn test_diagflat_cl() {
    let device = CLDevice::new(0).unwrap().select();

    let x = Matrix::from((&device, (1, 4), [1.5, 2., 6., 4.]));
    let result = device.diagflat(&x);
    assert_eq!(
        result.read(),
        vec![1.5, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 4.0]
    );

    //    println!("res: {:?}", result);
}


#[test]
fn test_diagflat_kernel_cl() {
    let device = CLDevice::new(0).unwrap().select();

    let x = Matrix::from((&device, (2, 4), [1.5, 2., 6., 4.,
                                                         5., 7., 3., 1.,]));
    let out = cl_diagflat(&device, &x).unwrap();
    let out = Matrix::from((out, x.rows(), x.cols()*x.cols()));
    println!("out: {out:?}");

    /* 
    assert_eq!(
        result.read(),
        vec![1.5, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 4.0]
    );
    */
}
