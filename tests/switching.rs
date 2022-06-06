use custos::{CLDevice, Matrix, InternCLDevice, GenericOCL, CPU, VecRead, AsDev, opencl::{api::{create_buffer, MemFlags, release_mem_object}, CL_CACHE, OclPtr}, Buffer, range};
use custos_math::FnsOps;


#[test]
fn test_unified_mem_device_switch() -> custos::Result<()> {
    let device = CLDevice::get(0)?.select();

    let a = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.,]));

    let start = std::time::Instant::now();
    for _ in range(10000000) {
        let m = cpu_exec(&device, a, |cpu, m| cpu.ln(&m))?;
    }

    println!("duration: {:?}", start.elapsed());

    let m = cpu_exec(&device, a, |cpu, m| cpu.ln(&m))?;
    println!("m: {m:?}");
    Ok(())
}

pub fn cpu_exec<T, F>(device: &InternCLDevice, matrix: Matrix<T>, f: F) -> custos::Result<Matrix<T>> 
where 
    F: Fn(&custos::InternCPU, Matrix<T>) -> Matrix<T>,
    T: GenericOCL
{
    let cpu = CPU::new();

    if device.unified_mem() && !cfg!(feature="safe"){

        // host ptr matrix
        let no_drop = f(&cpu, matrix);

        // use the host pointer to create an OpenCL buffer
        let cl_ptr = create_buffer(&device.ctx(), MemFlags::MemReadWrite | MemFlags::MemUseHostPtr, matrix.size(), Some(&no_drop))?;

        let old = CL_CACHE.with(|cache| {
            // add created buffer to the "caching chain"
            cache.borrow_mut().nodes.insert(custos::Node::new(matrix.size()), (OclPtr(cl_ptr), matrix.size()))
        });

        if let Some(old) = old {
            unsafe {
                release_mem_object(old.0.0)?;
            }
        }
        
        // TODO: When should the buffer be freed, if the "safe" feature is used?

        // Both lines prevent the deallocation of the underlying buffer.
        //Box::into_raw(Box::new(no_drop)); // "safe" mode
        // TODO: Deallocate cpu buffer? This may leak memory.
        cpu.cpu.borrow_mut().ptrs.clear(); // default mode
        
        let buf = Buffer {
            ptr: (no_drop.ptr.0, cl_ptr),
            len: matrix.size(),
        };
        let matrix = Matrix::from((buf, matrix.dims()));
        return Ok(matrix);
    }
    
    let x = if device.unified_mem() {
        matrix
    } else {
        // Read buffer that is allocated on an OpenCL device and create a new cpu matrix.
        Matrix::from((&cpu, matrix.dims(), device.read(matrix.as_buf())))
    };
    
    Ok(Matrix::from((device, f(&cpu, x))))
}

