use core::ffi::c_void;

use half::f16;

pub trait KernelParam {
    fn to_launch_arg(&self) -> *mut c_void {
        self as *const _ as *mut _
    }
}

impl<T: KernelParam> KernelParam for *mut T {}
impl<T: KernelParam> KernelParam for *const T {}
impl KernelParam for f16 {}
impl KernelParam for f32 {}
impl KernelParam for f64 {}
impl KernelParam for u8 {}
impl KernelParam for u16 {}
impl KernelParam for u32 {}
impl KernelParam for u64 {}
impl KernelParam for i8 {}
impl KernelParam for i16 {}
impl KernelParam for i32 {}
impl KernelParam for i64 {}
