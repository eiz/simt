#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

use std::{
    fmt::{Display, Formatter},
    sync::Once,
};

// bindgen /opt/rocm/include/hip/hip_runtime_api.h --no-layout-tests --default-enum-style rust --dynamic-loading hip --allowlist-function '^hip.*' -- -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ > src/bindings.rs
include!("bindings.rs");

static mut HIP_LIBRARY: Option<hip> = None;
static mut ONCE: Once = Once::new();

impl std::error::Error for hipError_t {}
impl Display for hipError_t {
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
            InitializeError::PreviouslyFailed => write!(f, "HIP has previously failed to load"),
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
            const LIB: &str = "libamdhip64.so";
            #[cfg(windows)]
            const LIB: &str = "shrugemoji.dll";
            match hip::new(LIB) {
                Ok(lib) => HIP_LIBRARY = Some(lib),
                Err(err) => e = Some(InitializeError::LoadError(err)),
            }
        });

        if HIP_LIBRARY.is_some() {
            return Ok(());
        }

        if let Some(e) = e {
            return Err(e);
        }
    }

    Err(InitializeError::PreviouslyFailed)
}

pub fn library() -> &'static hip {
    initialize().expect("failed to load HIP");
    unsafe { HIP_LIBRARY.as_ref().unwrap() }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    #[test]
    fn test_initialize() {
        initialize().unwrap();
    }

    #[test]
    fn test_library() {
        unsafe {
            let lib = library();
            assert_eq!(
                CStr::from_ptr(lib.hipGetErrorName(hipError_t::hipSuccess)),
                CStr::from_ptr(b"hipSuccess\0".as_ptr() as *const i8)
            );
        }
    }
}
