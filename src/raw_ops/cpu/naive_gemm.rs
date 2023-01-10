use core::ops::{AddAssign, Mul};

pub fn naive_gemm<T>(m: usize, k: usize, n: usize, a: &[T], b: &[T], c: &mut [T])
where
    T: Mul<Output = T> + Copy + Default + AddAssign,
{
    for row in 0..m {
        for col in 0..n {
            let a_row = &a[row * k..row * k + k];
            let mut acc = T::default();
            for elem in 0..k {
                acc += a_row[elem] * b[elem * n + col];
            }
            c[row * n + col] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn roughly_equals(lhs: &[f64], rhs: &[f64], diff: f64) {
        for (a, b) in lhs.iter().zip(rhs) {
            let abs = (*a - *b).abs();
            if abs > diff {
                panic!(
                    "\n left: '{:?}',\n right: '{:?}', \n left elem.: {} != right elem. {}",
                    lhs, rhs, a, b
                )
            }
        }
    }

    #[test]
    fn test_gemm1() {
        let a = [1f64, 4., 2., 9.];
        let b = [5., 4., 2., 9.];

        let m = 1;
        let k = 4;
        let n = 1;

        let mut c = [0.];

        naive_gemm(m, k, n, &a, &b, &mut c);

        assert_eq!(c[0], 106.);
    }

    #[test]
    fn test_larger_gemm1() {
        //5x7
        let arr1 = [
            9., 1., 3., 6., 7., 3., 63f64, 93., 51., 23., 36., 87., 3., 63., 9., 1., 43., 46.3, 7.,
            3., 63., 9., 15., 73., 6.3, 7., 53., 63., 69., 1., 3., 6., 7., 43., 63.,
        ];

        //7x10
        let arr2 = [
            1., 2., 3., 44., 55., 6., 7., 8., 95., 103., 14., 2., 33., 4., 75., 6., 37., 8., 9.,
            120., 31., 2., 3., 4., 5., 6.51, 7.45, 8., 9., 10., 313., 244., 3., 4., 5.8, 6., 27.,
            48., 9., 101., 21., 2., 3.4324, 4., 5., 6., 75., 38., 9., 109., 11., 2., 3., 4., 85.,
            96., 7., 8., 29., 130., 1., 2.91, 3.909, 4., 5.634, 36., 7., 8., 9., 130.,
        ];

        let mut out = [0.; 50];

        let _should = [
            2237.0, 1693.33, 366.2938, 728.0, 1264.742, 2713.53, 1271.35, 1186.0, 1662.0, 11026.0,
            14711.0, 9481.33, 2692.886, 5144.0, 10308.742, 4307.73, 10668.35, 6898.0, 11262.0,
            37628.0, 16090.899, 11606.53, 607.1938, 1049.2, 1698.482, 3215.73, 2657.45, 3440.4,
            2384.7, 15496.3, 5246.9, 2034.53, 1189.1938, 1265.2, 6916.482, 8055.0303, 2668.95,
            2272.4, 3870.7, 19936.3, 2737.0, 1893.33, 666.29376, 3528.0, 7964.7417, 6913.5303,
            1971.35, 1986.0, 8522.0, 22406.0,
        ];

        naive_gemm(5, 7, 10, &arr1, &arr2, &mut out);

        roughly_equals(&out, &_should, 0.01);
    }
}
