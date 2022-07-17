use custos::{CPU, AsDev};
use custos_math::{correlate_valid_mut, rot_kernel, add_full_padding, Matrix};

#[test]
fn test_correlate_slice() {
    let lhs = [
        1., 2., 3., 4., 
        5., 6., 7., 8., 
        9., 1., 2., 3., 
        4., 5., 6., 7.,
    ];

    let kernel = [1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let mut out = vec![0.; 4];
    correlate_valid_mut(&lhs, (4, 4), &kernel, (3, 3), &mut out);
    assert_eq!(out, vec![195., 177., 213., 222.]);

    let lhs = vec![4f32; 128 * 128];
    let kernel = vec![3f32; 3 * 3];
    let mut out = vec![0.; (128 - 3 + 1) * (128 - 3 + 1)];
    for _ in 0..100 {
        correlate_valid_mut(&lhs, (128, 128), &kernel, (3, 3), &mut out);
    }
}

#[test]
fn test_rotate_kernel() {
    let kernel = [1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let rot = rot_kernel(&kernel, (3, 3));
    assert_eq!(rot, vec![9., 8., 7., 6., 5., 4., 3., 2., 1.,]);
}

#[test]
fn test_add_padding() {
    let device = CPU::new().select();
    let lhs = [
        1., 2., 3., 4., 
        5., 6., 7., 8., 
        9., 1., 2., 3., 
        4., 5., 6., 7.,
    ];

    let out = add_full_padding(&lhs, (4, 4), (3, 3));
    assert_eq!(vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 
        0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 
        0.0, 0.0, 9.0, 1.0, 2.0, 3.0, 0.0, 0.0, 
        0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], out.0);
    let out = Matrix::from((&device, out.1, out.2, out.0));
    println!("out: {out:?}");
}