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

    #[error("failed to initialize cuda: {0}")]
    Initialize(#[from] simt_cuda_sys::InitializeError),
    #[error("cuda error '{0}'")]
    Cuda(#[from] simt_cuda_sys::CUresult),
}

type Result<T> = core::result::Result<T, Error>;

#[inline(always)]
pub unsafe fn cuda_call<F: FnOnce() -> simt_cuda_sys::CUresult>(cb: F) -> Result<()> {
    let res = cb();
    if res == simt_cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda(res))
    }
}

#[inline(always)]
pub unsafe fn cuda_result_call<T, F: FnOnce(*mut T) -> simt_cuda_sys::CUresult>(
    cb: F,
) -> Result<T> {
    let mut out = std::mem::MaybeUninit::uninit();
    let res = cb(out.as_mut_ptr());
    if res == simt_cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(out.assume_init())
    } else {
        Err(Error::Cuda(res))
    }
}

fn cuda_ensure_initialized() -> Result<()> {
    static ONCE: Once = Once::new();
    let mut init_err = None;
    unsafe {
        ONCE.call_once(|| {
            if let Err(e) = simt_cuda_sys::initialize() {
                init_err = Some(Error::Initialize(e));
            } else if let Err(e) = cuda_call(|| simt_cuda_sys::library().cuInit(0)) {
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
pub struct CudaPhysicalDevice(i32);

impl CudaPhysicalDevice {
    pub fn count() -> Result<i32> {
        cuda_ensure_initialized()?;
        unsafe { Ok(cuda_result_call(|x| simt_cuda_sys::library().cuDeviceGetCount(x))? as i32) }
    }

    pub fn index(&self) -> i32 {
        self.0
    }

    pub fn get(index: i32) -> Result<Self> {
        cuda_ensure_initialized()?;
        unsafe {
            Ok(CudaPhysicalDevice(cuda_result_call(|x| {
                simt_cuda_sys::library().cuDeviceGet(x, index)
            })?))
        }
    }

    pub fn name(&self) -> Result<String> {
        unsafe {
            let mut name = [0u8; 256];
            cuda_call(|| {
                simt_cuda_sys::library().cuDeviceGetName(
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
            let cuda = simt_cuda_sys::library();
            let major = cuda_result_call(|x| {
                cuda.cuDeviceGetAttribute(
                    x,
                    simt_cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    self.0,
                )
            })?;
            let minor = cuda_result_call(|x| {
                cuda.cuDeviceGetAttribute(
                    x,
                    simt_cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    self.0,
                )
            })?;
            Ok(major * 100 + minor)
        }
    }
}

pub struct CudaDevice {
    context: simt_cuda_sys::CUcontext,
}

impl CudaDevice {
    pub fn new(device: CudaPhysicalDevice) -> Result<Self> {
        unsafe {
            let cuda = simt_cuda_sys::library();
            let context = cuda_result_call(|x| cuda.cuCtxCreate_v2(x, 0, device.0))?;
            cuda_result_call(|x| cuda.cuCtxPopCurrent_v2(x))?;
            Ok(Self { context })
        }
    }

    pub fn lock(self: &Arc<Self>) -> Result<ScopedCudaDevice> {
        ScopedCudaDevice::new(self.clone())
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        unsafe {
            cuda_call(|| simt_cuda_sys::library().cuCtxDestroy_v2(self.context))
                .expect("cuCtxDestroy_v2 failed");
        }
    }
}

// exercise for the reader: make get() not have to return a clone
thread_local! {
    static THREAD_CUDA_CONTEXT: RefCell<Option<Arc<CudaDevice>>> = RefCell::new(None);
}

pub struct ScopedCudaDevice {
    old_value: Option<Arc<CudaDevice>>,
}

impl ScopedCudaDevice {
    fn new(value: Arc<CudaDevice>) -> Result<Self> {
        let cuda = simt_cuda_sys::library();
        let old_value = THREAD_CUDA_CONTEXT.with(|v| -> Result<Option<Arc<CudaDevice>>> {
            let mut v = v.borrow_mut();
            let old_value = v.clone();
            unsafe { cuda_call(|| cuda.cuCtxPushCurrent_v2(value.context))? }
            *v = Some(value);
            Ok(old_value)
        })?;
        Ok(Self { old_value })
    }

    pub fn get() -> Result<Arc<CudaDevice>> {
        THREAD_CUDA_CONTEXT
            .with(|v| v.borrow().clone())
            .ok_or(Error::Cuda(
                simt_cuda_sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
            ))
    }
}

impl Drop for ScopedCudaDevice {
    fn drop(&mut self) {
        THREAD_CUDA_CONTEXT.with(|v| {
            let mut v = v.borrow_mut();
            let cuda = simt_cuda_sys::library();
            unsafe {
                cuda_result_call(|x| cuda.cuCtxPopCurrent_v2(x))
                    .expect("cuCtxPopCurrent_v2 failed");
            }
            *v = self.old_value.clone()
        });
    }
}

pub struct CudaBuffer {
    pub ptr: simt_cuda_sys::CUdeviceptr_v2,
    pub size: usize,
}

impl CudaBuffer {
    pub fn new(size: usize) -> Result<Self> {
        let cuda = simt_cuda_sys::library();
        let ptr = unsafe { cuda_result_call(|x| cuda.cuMemAlloc_v2(x, size))? };
        Ok(Self { ptr, size })
    }

    pub unsafe fn copy_from(&mut self, src: *const std::ffi::c_void, size: usize) -> Result<()> {
        let cuda = simt_cuda_sys::library();
        cuda_call(|| cuda.cuMemcpyHtoD_v2(self.ptr, src as *mut _, size))?;
        Ok(())
    }

    pub unsafe fn copy_to(&self, dst: *mut std::ffi::c_void, size: usize) -> Result<()> {
        let cuda = simt_cuda_sys::library();
        cuda_call(|| cuda.cuMemcpyDtoH_v2(dst, self.ptr, size))?;
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

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            let cuda = simt_cuda_sys::library();
            cuda_call(|| cuda.cuMemFree_v2(self.ptr)).expect("cuMemFree_v2 failed");
        }
    }
}

pub struct CudaModule {
    inner: simt_cuda_sys::CUmodule,
}

impl CudaModule {
    pub unsafe fn new(data: &[u8]) -> Result<Self> {
        let cuda = simt_cuda_sys::library();
        let inner = cuda_result_call(|x| cuda.cuModuleLoadData(x, data.as_ptr() as *const _))?;
        Ok(Self { inner })
    }

    pub fn find(capability: i32, kernels: &[(&str, &[u8])]) -> Result<Self> {
        let cuda = simt_cuda_sys::library();
        let mut compatible_kernels = vec![];
        for (arch, bin) in kernels {
            if !arch.starts_with("sm") {
                continue;
            }
            let arch = arch[2..].parse::<i32>().unwrap_or(0x7FFF_FFFF);
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
            unsafe { cuda_result_call(|x| cuda.cuModuleLoadData(x, bin.as_ptr() as *const _))? };
        Ok(Self { inner })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        unsafe {
            let cuda = simt_cuda_sys::library();
            cuda_call(|| cuda.cuModuleUnload(self.inner)).expect("cuModuleUnload failed");
        }
    }
}

pub struct CudaStream {
    inner: simt_cuda_sys::CUstream,
}

impl CudaStream {
    pub fn new() -> Result<Self> {
        let cuda = simt_cuda_sys::library();
        // TODO: create nonblocking streams
        let inner = unsafe { cuda_result_call(|x| cuda.cuStreamCreate(x, 0))? };
        Ok(Self { inner })
    }

    pub fn sync(&self) -> Result<()> {
        let cuda = simt_cuda_sys::library();
        unsafe { cuda_call(|| cuda.cuStreamSynchronize(self.inner))? }
        Ok(())
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            let cuda = simt_cuda_sys::library();
            cuda_call(|| cuda.cuStreamDestroy_v2(self.inner)).expect("cuStreamDestroy_v2 failed");
        }
    }
}

#[derive(Default, Clone)]
pub struct LaunchParams<'a> {
    pub blocks: (u32, u32, u32),
    pub threads: (u32, u32, u32),
    pub shared_mem: u32,
    pub stream: Option<&'a CudaStream>,
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
    ptr: simt_cuda_sys::CUfunction,
    _dead: PhantomData<T>,
}

impl<T> Kernel<T> {
    pub fn new(module: &CudaModule, name: &str) -> Result<Self> {
        let cuda = simt_cuda_sys::library();
        let c_name = CString::new(name).map_err(|_| Error::InvalidKernelName)?;

        unsafe {
            let result = cuda_result_call(|x| {
                cuda.cuModuleGetFunction(x, module.inner, c_name.as_ptr() as *const i8)
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
                let cuda = simt_cuda_sys::library();
                let (bx, by, bz) = launch_params.blocks;
                let (tx, ty, tz) = launch_params.threads;
                unsafe {
                    cuda_call(|| {
                        cuda.cuLaunchKernel(
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
