
fn main() {

    println!("cargo:rerun-if-env-changed=CUSTOS_CL_DEVICE_IDX");
    println!("cargo:rerun-if-env-changed=CUSTOS_CU_DEVICE_IDX");
    println!("cargo:rerun-if-env-changed=CUSTOS_USE_UNIFIED");

    #[cfg(feature="opencl")]
    if custos::UNIFIED_CL_MEM {
        println!("cargo:rustc-cfg=unified_cl");
    }   

}