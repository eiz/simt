use alloc::{ffi::CString, sync::Arc};
use core::{
    cell::RefCell,
    ffi::{c_void, CStr},
    marker::PhantomData,
};
use half::f16;
use std::sync::Once;
use thiserror::Error;

extern crate alloc;

#[derive(Error, Debug)]
pub enum Error {
    #[error("no compatible kernel found")]
    NoCompatibleKernel,
    #[error("invalid kernel name")]
    InvalidKernelName,
    #[error("invalid utf8")]
    InvalidUtf8,

    #[error("failed to initialize hip: {0}")]
    Initialize(#[from] simt_hip_sys::InitializeError),
    #[error("hip error '{0}'")]
    Hip(#[from] simt_hip_sys::hipError_t),
}

type Result<T> = core::result::Result<T, Error>;

#[inline(always)]
pub unsafe fn hip_call<F: FnOnce() -> simt_hip_sys::hipError_t>(cb: F) -> Result<()> {
    let res = cb();
    if res == simt_hip_sys::hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(Error::Hip(res))
    }
}

#[inline(always)]
pub unsafe fn hip_result_call<T, F: FnOnce(*mut T) -> simt_hip_sys::hipError_t>(
    cb: F,
) -> Result<T> {
    let mut out = std::mem::MaybeUninit::uninit();
    let res = cb(out.as_mut_ptr());
    if res == simt_hip_sys::hipError_t::hipSuccess {
        Ok(out.assume_init())
    } else {
        Err(Error::Hip(res))
    }
}

fn hip_ensure_initialized() -> Result<()> {
    static ONCE: Once = Once::new();
    let mut init_err = None;
    unsafe {
        ONCE.call_once(|| {
            if let Err(e) = simt_hip_sys::initialize() {
                init_err = Some(Error::Initialize(e));
            } else if let Err(e) = hip_call(|| simt_hip_sys::library().hipInit(0)) {
                init_err = Some(e);
            }
        });
    }
    if init_err.is_some() {
        Err(init_err.unwrap())
    } else {
        Ok(())
    }
}

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub struct HipPhysicalDevice(i32);

impl HipPhysicalDevice {
    pub fn count() -> Result<i32> {
        hip_ensure_initialized()?;
        unsafe { Ok(hip_result_call(|x| simt_hip_sys::library().hipGetDeviceCount(x))? as i32) }
    }

    pub fn index(&self) -> i32 {
        self.0
    }

    pub fn get(index: i32) -> Result<Self> {
        hip_ensure_initialized()?;
        unsafe {
            Ok(HipPhysicalDevice(hip_result_call(|x| {
                simt_hip_sys::library().hipDeviceGet(x, index)
            })?))
        }
    }

    pub fn name(&self) -> Result<String> {
        unsafe {
            let mut name = [0u8; 256];
            hip_call(|| {
                simt_hip_sys::library().hipDeviceGetName(
                    name.as_mut_ptr() as *mut _,
                    name.len() as i32,
                    self.0,
                )
            })?;
            Ok(CStr::from_ptr(name.as_ptr() as *const i8)
                .to_str()
                .map_err(|_| Error::InvalidUtf8)?
                .to_owned())
        }
    }

    pub fn capability(&self) -> Result<i32> {
        unsafe {
            let hip = simt_hip_sys::library();
            let major = hip_result_call(|x| {
                hip.hipDeviceGetAttribute(
                    x,
                    simt_hip_sys::hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMajor,
                    self.0,
                )
            })?;
            let minor = hip_result_call(|x| {
                hip.hipDeviceGetAttribute(
                    x,
                    simt_hip_sys::hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMinor,
                    self.0,
                )
            })?;
            Ok(major * 100 + minor)
        }
    }
}

pub struct HipDevice {
    context: simt_hip_sys::hipCtx_t,
}

impl HipDevice {
    pub fn new(device: HipPhysicalDevice) -> Result<Self> {
        unsafe {
            let hip = simt_hip_sys::library();
            let context = hip_result_call(|x| hip.hipCtxCreate(x, 0, device.0))?;
            hip_result_call(|x| hip.hipCtxPopCurrent(x)).expect("hipCtxPopCurrent failed");
            Ok(Self { context })
        }
    }

    pub fn lock(self: &Arc<Self>) -> Result<ScopedHipDevice> {
        ScopedHipDevice::new(self.clone())
    }
}

impl Drop for HipDevice {
    fn drop(&mut self) {
        unsafe {
            hip_call(|| simt_hip_sys::library().hipCtxDestroy(self.context))
                .expect("hipCtxDestroy failed");
        }
    }
}

// exercise for the reader: make get() not have to return a clone
thread_local! {
    static THREAD_HIP_CONTEXT: RefCell<Option<Arc<HipDevice>>> = RefCell::new(None);
}

pub struct ScopedHipDevice {
    old_value: Option<Arc<HipDevice>>,
}

impl ScopedHipDevice {
    fn new(value: Arc<HipDevice>) -> Result<Self> {
        let hip = simt_hip_sys::library();
        let old_value = THREAD_HIP_CONTEXT.with(|v| -> Result<Option<Arc<HipDevice>>> {
            let mut v = v.borrow_mut();
            let old_value = v.clone();
            unsafe { hip_call(|| hip.hipCtxPushCurrent(value.context))? }
            *v = Some(value);
            Ok(old_value)
        })?;
        Ok(Self { old_value })
    }

    pub fn get() -> Result<Arc<HipDevice>> {
        THREAD_HIP_CONTEXT
            .with(|v| v.borrow().clone())
            .ok_or(Error::Hip(simt_hip_sys::hipError_t::hipErrorInvalidContext))
    }
}

impl Drop for ScopedHipDevice {
    fn drop(&mut self) {
        THREAD_HIP_CONTEXT.with(|v| {
            let mut v = v.borrow_mut();
            let hip = simt_hip_sys::library();
            unsafe {
                hip_result_call(|x| hip.hipCtxPopCurrent(x)).expect("hipCtxPopCurrent failed");
            }
            *v = self.old_value.clone()
        });
    }
}

pub struct HipBuffer {
    pub ptr: simt_hip_sys::hipDeviceptr_t,
    pub size: usize,
}

impl HipBuffer {
    pub fn new(size: usize) -> Result<Self> {
        let hip = simt_hip_sys::library();
        let ptr = unsafe { hip_result_call(|x| hip.hipMalloc(x, size))? };
        Ok(Self { ptr, size })
    }

    pub unsafe fn copy_from(&mut self, src: *const std::ffi::c_void, size: usize) -> Result<()> {
        let hip = simt_hip_sys::library();
        hip_call(|| hip.hipMemcpyHtoD(self.ptr, src as *mut _, size))?;
        Ok(())
    }

    pub unsafe fn copy_to(&self, dst: *mut std::ffi::c_void, size: usize) -> Result<()> {
        let hip = simt_hip_sys::library();
        hip_call(|| hip.hipMemcpyDtoH(dst, self.ptr, size))?;
        Ok(())
    }

    pub fn copy_from_slice<T: Copy>(&mut self, src: &[T]) -> Result<()> {
        assert_eq!(src.len() * std::mem::size_of::<T>(), self.size);
        unsafe { self.copy_from(src.as_ptr() as *const std::ffi::c_void, self.size) }
    }

    pub fn copy_to_slice<T: Copy>(&self, dst: &mut [T]) -> Result<()> {
        assert_eq!(dst.len() * std::mem::size_of::<T>(), self.size);
        unsafe { self.copy_to(dst.as_mut_ptr() as *mut std::ffi::c_void, self.size) }
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        unsafe {
            let hip = simt_hip_sys::library();
            hip_call(|| hip.hipFree(self.ptr)).expect("hipFree failed");
        }
    }
}

pub struct HipModule {
    inner: simt_hip_sys::hipModule_t,
}

impl HipModule {
    pub unsafe fn new(data: &[u8]) -> Result<Self> {
        let hip = simt_hip_sys::library();
        let inner = hip_result_call(|x| hip.hipModuleLoadData(x, data.as_ptr() as *const _))?;
        Ok(Self { inner })
    }

    pub fn find(capability: i32, kernels: &[(&str, &[u8])]) -> Result<Self> {
        let hip = simt_hip_sys::library();
        let mut compatible_kernels = vec![];
        for (arch, bin) in kernels {
            if !arch.starts_with("gfx") {
                continue;
            }
            let arch = arch[3..].parse::<i32>().unwrap_or(0x7FFF_FFFF);
            if arch <= capability {
                compatible_kernels.push((arch, bin));
            }
        }
        compatible_kernels.sort_by_key(|(arch, _)| *arch);
        let (_, bin) = compatible_kernels
            .iter()
            .rev()
            .filter(|(arch, _)| *arch <= capability)
            .last()
            .ok_or_else(|| Error::NoCompatibleKernel)?;
        let inner =
            unsafe { hip_result_call(|x| hip.hipModuleLoadData(x, bin.as_ptr() as *const _))? };
        Ok(Self { inner })
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        unsafe {
            let hip = simt_hip_sys::library();
            hip_call(|| hip.hipModuleUnload(self.inner)).expect("hipModuleUnload failed");
        }
    }
}

pub struct HipStream {
    inner: simt_hip_sys::hipStream_t,
}

impl HipStream {
    pub fn new() -> Result<Self> {
        let hip = simt_hip_sys::library();
        let inner = unsafe { hip_result_call(|x| hip.hipStreamCreate(x))? };
        Ok(Self { inner })
    }

    pub fn sync(&self) -> Result<()> {
        let hip = simt_hip_sys::library();
        unsafe { hip_call(|| hip.hipStreamSynchronize(self.inner))? }
        Ok(())
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        unsafe {
            let hip = simt_hip_sys::library();
            hip_call(|| hip.hipStreamDestroy(self.inner)).expect("hipStreamDestroy failed");
        }
    }
}

#[derive(Default, Clone)]
pub struct LaunchParams<'a> {
    pub blocks: (u32, u32, u32),
    pub threads: (u32, u32, u32),
    pub shared_mem: u32,
    pub stream: Option<&'a HipStream>,
}

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

pub struct Kernel<T> {
    ptr: simt_hip_sys::hipFunction_t,
    _dead: PhantomData<T>,
}

impl<T> Kernel<T> {
    pub fn new(module: &HipModule, name: &str) -> Result<Self> {
        let hip = simt_hip_sys::library();
        let c_name = CString::new(name).map_err(|_| Error::InvalidKernelName)?;

        unsafe {
            let result = hip_result_call(|x| {
                hip.hipModuleGetFunction(x, module.inner, c_name.as_ptr() as *const i8)
            })?;

            Ok(Self {
                ptr: result,
                _dead: PhantomData,
            })
        }
    }
}

// TODO: launch should really be declared unsafe, but 'ehhh
// that would be mildly inconvenient
macro_rules! impl_kernel {
    (($($ty_param:ident),*), ($($ty_idx:tt),*)) => {
        impl<$($ty_param: KernelParam),*> Kernel<($($ty_param),*,)> {
            pub fn launch(
                &self,
                launch_params: LaunchParams,
                params: ($($ty_param),*,),
            ) -> Result<()> {
                let hip = simt_hip_sys::library();
                let (bx, by, bz) = launch_params.blocks;
                let (tx, ty, tz) = launch_params.threads;
                unsafe {
                    hip_call(|| {
                        hip.hipModuleLaunchKernel(
                            self.ptr,
                            bx,
                            by,
                            bz,
                            tx,
                            ty,
                            tz,
                            launch_params.shared_mem,
                            launch_params.stream.map(|x| x.inner).unwrap_or(std::ptr::null_mut()),
                            &[$(params.$ty_idx.to_launch_arg()),*] as *const _ as *mut _,
                            std::ptr::null_mut(),
                        )
                    })?;
                }
                Ok(())
            }
        }
    };
}
impl_kernel!((T1), (0));
impl_kernel!((T1, T2), (0, 1));
impl_kernel!((T1, T2, T3), (0, 1, 2));
impl_kernel!((T1, T2, T3, T4), (0, 1, 2, 3));
impl_kernel!((T1, T2, T3, T4, T5), (0, 1, 2, 3, 4));
impl_kernel!((T1, T2, T3, T4, T5, T6), (0, 1, 2, 3, 4, 5));
impl_kernel!((T1, T2, T3, T4, T5, T6, T7), (0, 1, 2, 3, 4, 5, 6));
impl_kernel!((T1, T2, T3, T4, T5, T6, T7, T8), (0, 1, 2, 3, 4, 5, 6, 7));
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9),
    (0, 1, 2, 3, 4, 5, 6, 7, 8)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
);
impl_kernel!(
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
);
impl_kernel!(
    (
        T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20,
        T21
    ),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
);
impl_kernel!(
    (
        T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20,
        T21, T22
    ),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
);
impl_kernel!(
    (
        T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20,
        T21, T22, T23
    ),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
);
impl_kernel!(
    (
        T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20,
        T21, T22, T23, T24
    ),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
);
