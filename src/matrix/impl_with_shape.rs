use custos::{Buffer, Device, Dim2, WithShape};

use crate::Matrix;

impl<'a, T, D, C, const B: usize, const A: usize> WithShape<&'a D, C>
    for Matrix<'a, T, D, Dim2<B, A>>
where
    D: Device,
    Buffer<'a, T, D, Dim2<B, A>>: WithShape<&'a D, C>,
{
    fn with(device: &'a D, array: C) -> Self {
        let data = Buffer::with(device, array);
        Matrix { data, dims: (B, A) }
    }
}

//impl<'a, T, D>

/*
#[cfg(test)]
mod tests {

    #[cfg(feature = "cpu")]
    #[test]
    fn test_with() {
        use custos::{Dim2, WithShape, CPU};

        use crate::Matrix;

        let device = CPU::new();

        let mat = Matrix::<_, _, Dim2<1, 3>>::with(&device, [3, 2, 5]);
    }
}
*/
