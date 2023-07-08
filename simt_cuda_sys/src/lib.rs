#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

use std::{
    fmt::{Display, Formatter},
    sync::Once,
};

// bindgen /usr/local/cuda-12.0/include/cuda.h --no-layout-tests --default-enum-style rust --dynamic-loading cuda --allowlist-function '^cu.*' > src/bindings.rs
include!("bindings.rs");

static mut CUDA_LIBRARY: Option<cuda> = None;
static mut ONCE: Once = Once::new();

impl std::error::Error for CUresult {}
impl Display for CUresult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
pub enum InitializeError {
    PreviouslyFailed,
    LoadError(libloading::Error),
}

impl Display for InitializeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InitializeError::PreviouslyFailed => write!(f, "CUDA has previously failed to load"),
            InitializeError::LoadError(err) => write!(f, "{}", err),
        }
    }
}
impl std::error::Error for InitializeError {}

pub fn initialize() -> Result<(), InitializeError> {
    unsafe {
        let mut e = None;
        ONCE.call_once(|| {
            #[cfg(target_os = "linux")]
            const LIB: &str = "libcuda.so";
            #[cfg(windows)]
            const LIB: &str = "shrugemoji.dll";
            match cuda::new(LIB) {
                Ok(lib) => CUDA_LIBRARY = Some(lib),
                Err(err) => e = Some(InitializeError::LoadError(err)),
            }
        });

        if CUDA_LIBRARY.is_some() {
            return Ok(());
        }

        if let Some(e) = e {
            return Err(e);
        }

        Err(InitializeError::PreviouslyFailed)
    }
}

pub fn library() -> &'static cuda {
    initialize().expect("failed to load CUDA");
    unsafe { CUDA_LIBRARY.as_ref().unwrap() }
}
