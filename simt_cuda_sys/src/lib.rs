#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

use std::fmt::{Display, Formatter};

// bindgen /usr/local/cuda-12.0/include/cuda.h --no-layout-tests --default-enum-style rust --dynamic-loading cuda --allowlist-function '^cu.*' > src/bindings.rs
include!("bindings.rs");

impl std::error::Error for CUresult {}
impl Display for CUresult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
