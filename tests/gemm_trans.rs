use std::time::Instant;

use custos::{range, Cache, GenericBlas, CPU};
use custos_math::Matrix;

#[test]
fn test_gemm_trans() {
    let device = CPU::new();

    let mat = Matrix::<f32>::from((
        &device,
        4,
        3,
        [1., 4., 6., 3., 1., 7., 9., 4., 1., 5., 4., 3.],
    ));
    let trans_mat = mat.T();

    let out_t = mat.gemm(&trans_mat);

    let mut out = Matrix::new(&device, (4, 4));
    GenericBlas::gemmT(4, 4, 3, &mat, &mat, &mut out);

    assert_eq!(out_t.as_slice(), out.as_slice());
}

#[test]
fn test_gemm_trans_perf() {
    let device = CPU::new();

    let mat = Matrix::<f32>::from((&device, 100, 300, vec![1.; 100 * 300]));
    let start = Instant::now();

    for _ in range(0..10) {
        let trans_mat = mat.T();
        let _out_t = mat.gemm(&trans_mat);
    }
    println!("pre_trans elapsed: {:?}", start.elapsed());

    let start = Instant::now();

    for _ in range(0..10) {
        let mut out = Cache::get::<f32, 0>(&device, mat.rows() * mat.rows(), ());
        GenericBlas::gemmT(mat.rows(), mat.rows(), mat.cols(), &mat, &mat, &mut out);
    }

    println!("trans blas elapsed: {:?}", start.elapsed());

    let mut out: custos::Buffer<f32> = Cache::get(&device, mat.rows() * mat.rows(), ());
    GenericBlas::gemmT(mat.rows(), mat.rows(), mat.cols(), &mat, &mat, &mut out);

    let trans_mat = mat.T();
    let out_t = mat.gemm(&trans_mat);

    println!("");
    assert_eq!(out_t.as_slice(), out.as_slice());
}

// TODO: does not work
/*#[test]
fn test_trans_gemm() {
    let device = CPU::new();

    let mat = Matrix::<f32>::from((&device, 4, 3, [1., 4., 6.,
                                                                3., 1., 7.,
                                                                9., 4., 1.,
                                                                5., 4., 3.,]));
    let trans_mat = mat.T();
    let out_t = trans_mat.gemm(&mat);

    println!("out_t: {out_t:?}");

    let mut out = Matrix::new(&device, (3, 3));
    GenericBlas::Tgemm(4, 4, 3, &mat, &mat, &mut out);

    println!();

    println!("out_t blas: {out:?}");

    //assert_eq!(out_t.as_slice(), out.as_slice());
}*/
