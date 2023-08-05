use criterion::{criterion_group, criterion_main, Criterion};
use custos::{CUDA, Buffer, get_count, set_count, Device};
use custos_math::{add_to_row_cu, Matrix, add_to_row_cu_2dim};

const M: usize = 2048*16;
const N: usize = 128;


pub fn add_to_row_bench(ben: &mut Criterion) {
    let device = CUDA::new(0).unwrap();

    
    let lhs = Matrix::from((&device, M,N, vec![1.5; M*N]));
    let rhs = Matrix::from((&device, 1, N, vec![1.5; N]));

    let count = get_count();

    ben.bench_function("bench add_to_row", |bench| bench.iter(|| {
        let mut out = device.retrieve(M*N, (&*lhs, &*rhs));

        add_to_row_cu(&device, &lhs, M, N, &rhs, &mut out).unwrap();
        device.stream().sync().unwrap();

        unsafe { set_count(count) }

    }));
}

fn add_to_row_2dim_bench(ben: &mut Criterion) {
    let device = CUDA::new(0).unwrap();

    
    let lhs = Matrix::from((&device, M,N, vec![1.5; M*N]));
    let rhs = Matrix::from((&device, 1, N, vec![1.5; N]));

    let count = get_count();

    ben.bench_function("bench add_to2dim_row", |bench| bench.iter(|| {
        let mut out = device.retrieve(M*N, (&*lhs, &*rhs));

        add_to_row_cu_2dim(&device, &lhs, M, N, &rhs, &mut out).unwrap();
        device.stream().sync().unwrap();

        unsafe { set_count(count) }

    }));
}


fn add_to_row_cpu_bench(ben: &mut Criterion) {
    let device = CUDA::new(0).unwrap();
    
    let lhs = Matrix::from((&device, M,N, vec![1.5; M*N]));
    let rhs = Matrix::from((&device, 1, N, vec![1.5; N]));

    let count = get_count();

    ben.bench_function("bench add_to_row_cpu", |bench| bench.iter(|| {

        lhs.add_row(&rhs);
        unsafe { set_count(count) }
    }));
}


criterion_group!(benches, add_to_row_bench, add_to_row_cpu_bench, add_to_row_2dim_bench);
criterion_main!(benches);