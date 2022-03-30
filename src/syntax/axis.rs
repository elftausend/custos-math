use custos::Matrix;

#[derive(Debug, Clone, Copy)]
pub enum Axis {
    Rows,
    Cols,
    All
}

pub trait GetAxis<T> {
    fn get_axis(&self) -> (Matrix<T>, Axis);
}

impl <T: Copy>GetAxis<T> for (Matrix<T>, Axis) {
    fn get_axis(&self) -> (Matrix<T>, Axis) {
        *self
    }
}

impl <T: Copy>GetAxis<T> for Matrix<T> {
    fn get_axis(&self) -> (Matrix<T>, Axis) {
        (*self, Axis::All)
    }
}