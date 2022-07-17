use custos_math::correlate_valid_mut;

#[test]
fn test_correlate_slice() {
    let lhs = [
        1., 2., 3., 4.,
        5., 6., 7., 8.,
        9., 1., 2., 3.,
        4., 5., 6., 7.
    ];

    let kernel = [
        1., 2., 3.,
        4., 5., 6.,
        7., 8., 9.,
    ];
    let mut out = vec![0.; 4];
    correlate_valid_mut(&lhs, (4, 4), &kernel, (3, 3), &mut out);
    assert_eq!(out, vec![195., 177., 213., 222.]);

    let lhs = vec![4f32; 128*128];
    let kernel = vec![3f32; 3*3];
    let mut out = vec![0.; (128-3+1) * (128-3+1)];
    for _ in 0..56000 {
        correlate_valid_mut(&lhs, (128, 128), &kernel, (3, 3), &mut out);
    }
    
}