#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

use std::fmt::{Formatter, Display};

// bindgen /opt/rocm/include/hip/hip_runtime_api.h --no-layout-tests --default-enum-style rust --dynamic-loading hip --allowlist-function '^hip.*' -- -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ > src/bindings.rs
include!("bindings.rs");

impl std::error::Error for hipError_t {}
impl Display for hipError_t {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
