#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

// bindgen /opt/rocm/include/rocblas/rocblas.h --no-layout-tests --default-enum-style rust --dynamic-loading rocblas --allowlist-function '^rocblas.*' -- -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ > bindings.rs
include!("bindings.rs");
