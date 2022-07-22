use custos::{CPU, CudaDevice, WriteBuf, VecRead};

use crate::Matrix;

pub fn cu_to_cpu_lr<T: Copy + Default, F: Fn(&CPU, &Matrix<T>, &Matrix<T>) -> Matrix<T>>(
    device: &CudaDevice,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    let cpu = custos::CPU::new();
    let lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs)));
    let rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs)));

    let result = f(&cpu, &lhs, &rhs);
    Matrix::from((device, result))
}

pub fn cu_to_cpu_lr_mut<T: Copy + Default, F: Fn(&CPU, &mut Matrix<T>, &Matrix<T>)>(
    device: &CudaDevice,
    lhs: &mut Matrix<T>,
    rhs: &Matrix<T>,
    f: F) 
{
    let cpu = custos::CPU::new();
    let mut cpu_lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs)));
    let cpu_rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs)));

    f(&cpu, &mut cpu_lhs, &cpu_rhs);
    device.write(lhs, &cpu_lhs);
}

pub fn cu_to_cpu_s<T: Copy + Default, F: Fn(&CPU, Matrix<T>) -> Matrix<T>>(
    device: &CudaDevice,
    x: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), device.read(x)));

    let result = f(&cpu, x);
    Matrix::from((device, result))
}

pub fn cu_to_cpu_s_mut<T: Copy + Default, F: Fn(&CPU, &mut Matrix<T>)>(
    device: &CudaDevice,
    x: &mut Matrix<T>,
    f: F) 
{
    let cpu = custos::CPU::new();
    let mut cpux = Matrix::from((&cpu, x.dims(), device.read(x)));

    f(&cpu, &mut cpux);
    device.write(x, &cpux)
}

pub fn cu_to_cpu_scalar<T: Copy + Default, F: Fn(&CPU, Matrix<T>) -> T>(
    device: &CudaDevice,
    x: &Matrix<T>,
    f: F,
) -> T {
    let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), device.read(x)));
    f(&cpu, x)
}
