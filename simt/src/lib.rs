use alloc::sync::Arc;
use core::cell::Cell;

use simt_core::KernelParam;
use simt_cuda::{CudaDevice, CudaKernel, CudaLaunchParams, CudaPhysicalDevice, ScopedCudaDevice};
use simt_hip::{HipDevice, HipKernel, HipLaunchParams, HipPhysicalDevice, ScopedHipDevice};
use thiserror::Error;

extern crate alloc;

#[derive(Error, Debug)]
pub enum Error {
    #[error("no compatible kernel found")]
    NoCompatibleKernel,
    #[error("no current device found")]
    NoCurrentDevice,
    #[error("invalid kernel name")]
    InvalidKernelName,
    #[error("invalid utf8")]
    InvalidUtf8,
    #[error("failed to initialize cuda: {0}")]
    InitializeCuda(#[from] simt_cuda_sys::InitializeError),
    #[error("failed to initialize hip: {0}")]
    InitializeHip(#[from] simt_hip_sys::InitializeError),
    #[error("cuda error '{0}'")]
    Cuda(#[from] simt_cuda_sys::CUresult),
    #[error("hip error '{0}'")]
    Hip(#[from] simt_hip_sys::hipError_t),
}

impl From<simt_cuda::Error> for Error {
    fn from(error: simt_cuda::Error) -> Self {
        match error {
            simt_cuda::Error::NoCompatibleKernel => Error::NoCompatibleKernel,
            simt_cuda::Error::InvalidKernelName => Error::InvalidKernelName,
            simt_cuda::Error::InvalidUtf8 => Error::InvalidUtf8,
            simt_cuda::Error::Initialize(e) => Error::InitializeCuda(e),
            simt_cuda::Error::Cuda(e) => Error::Cuda(e),
        }
    }
}

impl From<simt_hip::Error> for Error {
    fn from(error: simt_hip::Error) -> Self {
        match error {
            simt_hip::Error::NoCompatibleKernel => Error::NoCompatibleKernel,
            simt_hip::Error::InvalidKernelName => Error::InvalidKernelName,
            simt_hip::Error::InvalidUtf8 => Error::InvalidUtf8,
            simt_hip::Error::Initialize(e) => Error::InitializeHip(e),
            simt_hip::Error::Hip(e) => Error::Hip(e),
        }
    }
}

type Result<T> = core::result::Result<T, Error>;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ComputeApi {
    Cuda,
    Hip,
}

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub enum PhysicalGpu {
    Cuda(CudaPhysicalDevice),
    Hip(HipPhysicalDevice),
}

impl PhysicalGpu {
    pub fn any() -> Option<Self> {
        CudaPhysicalDevice::get(0)
            .ok()
            .map(Self::Cuda)
            .or_else(|| HipPhysicalDevice::get(0).ok().map(Self::Hip))
    }

    pub fn count(api: ComputeApi) -> Result<i32> {
        match api {
            ComputeApi::Cuda => Ok(CudaPhysicalDevice::count()?),
            ComputeApi::Hip => Ok(HipPhysicalDevice::count()?),
        }
    }

    pub fn index(&self) -> i32 {
        match self {
            Self::Cuda(device) => device.index(),
            Self::Hip(device) => device.index(),
        }
    }

    pub fn get(api: ComputeApi, index: i32) -> Result<Self> {
        match api {
            ComputeApi::Cuda => Ok(Self::Cuda(CudaPhysicalDevice::get(index)?)),
            ComputeApi::Hip => Ok(Self::Hip(HipPhysicalDevice::get(index)?)),
        }
    }

    pub fn name(&self) -> Result<String> {
        match self {
            Self::Cuda(device) => Ok(device.name()?),
            Self::Hip(device) => Ok(device.name()?),
        }
    }

    pub fn capability(&self) -> Result<i32> {
        match self {
            Self::Cuda(device) => Ok(device.capability()?),
            Self::Hip(device) => Ok(device.capability()?),
        }
    }
}

impl From<CudaPhysicalDevice> for PhysicalGpu {
    fn from(device: CudaPhysicalDevice) -> Self {
        Self::Cuda(device)
    }
}

impl From<HipPhysicalDevice> for PhysicalGpu {
    fn from(device: HipPhysicalDevice) -> Self {
        Self::Hip(device)
    }
}

#[derive(Clone)]
pub enum Gpu {
    Cuda(Arc<CudaDevice>),
    Hip(Arc<HipDevice>),
}

impl Gpu {
    pub fn new(device: PhysicalGpu) -> Result<Self> {
        match device {
            PhysicalGpu::Cuda(device) => Ok(Self::Cuda(Arc::new(CudaDevice::new(device)?))),
            PhysicalGpu::Hip(device) => Ok(Self::Hip(Arc::new(HipDevice::new(device)?))),
        }
    }

    pub fn lock(self: &Arc<Self>) -> Result<ScopedGpu> {
        match &**self {
            Self::Cuda(device) => Ok(device.lock()?.into()),
            Self::Hip(device) => Ok(device.lock()?.into()),
        }
    }
}

impl From<CudaDevice> for Gpu {
    fn from(device: CudaDevice) -> Self {
        Self::Cuda(Arc::new(device))
    }
}

impl From<HipDevice> for Gpu {
    fn from(device: HipDevice) -> Self {
        Self::Hip(Arc::new(device))
    }
}

thread_local! {
    pub static CURRENT_COMPUTE_API: Cell<Option<ComputeApi>> = Cell::new(None);
}

pub enum ScopedGpu {
    Cuda(Option<ComputeApi>, ScopedCudaDevice),
    Hip(Option<ComputeApi>, ScopedHipDevice),
}

impl ScopedGpu {
    fn push_api(api: ComputeApi) -> Option<ComputeApi> {
        CURRENT_COMPUTE_API.with(|current| {
            let previous = current.get();
            current.set(Some(api));
            previous
        })
    }

    pub fn current_api() -> Option<ComputeApi> {
        CURRENT_COMPUTE_API.with(|current| current.get())
    }
}

impl From<ScopedCudaDevice> for ScopedGpu {
    fn from(device: ScopedCudaDevice) -> Self {
        Self::Cuda(Self::push_api(ComputeApi::Cuda), device)
    }
}

impl From<ScopedHipDevice> for ScopedGpu {
    fn from(device: ScopedHipDevice) -> Self {
        Self::Hip(Self::push_api(ComputeApi::Hip), device)
    }
}

impl Drop for ScopedGpu {
    fn drop(&mut self) {
        let previous = match self {
            Self::Cuda(previous, _) => previous,
            Self::Hip(previous, _) => previous,
        };
        CURRENT_COMPUTE_API.with(|current| {
            current.set(*previous);
        })
    }
}

pub enum GpuBuffer {
    Cuda(simt_cuda::CudaBuffer),
    Hip(simt_hip::HipBuffer),
}

impl GpuBuffer {
    pub fn new(size: usize) -> Result<Self> {
        match ScopedGpu::current_api() {
            Some(ComputeApi::Cuda) => Ok(Self::Cuda(simt_cuda::CudaBuffer::new(size)?)),
            Some(ComputeApi::Hip) => Ok(Self::Hip(simt_hip::HipBuffer::new(size)?)),
            None => Err(Error::NoCurrentDevice),
        }
    }

    pub unsafe fn copy_from(&mut self, src: *const std::ffi::c_void, size: usize) -> Result<()> {
        match self {
            Self::Cuda(buffer) => Ok(buffer.copy_from(src, size)?),
            Self::Hip(buffer) => Ok(buffer.copy_from(src, size)?),
        }
    }

    pub unsafe fn copy_to(&self, dst: *mut std::ffi::c_void, size: usize) -> Result<()> {
        match self {
            Self::Cuda(buffer) => Ok(buffer.copy_to(dst, size)?),
            Self::Hip(buffer) => Ok(buffer.copy_to(dst, size)?),
        }
    }

    pub fn copy_from_slice<T: Copy>(&mut self, src: &[T]) -> Result<()> {
        match self {
            Self::Cuda(buffer) => Ok(buffer.copy_from_slice(src)?),
            Self::Hip(buffer) => Ok(buffer.copy_from_slice(src)?),
        }
    }

    pub fn copy_to_slice<T: Copy>(&self, dst: &mut [T]) -> Result<()> {
        match self {
            Self::Cuda(buffer) => Ok(buffer.copy_to_slice(dst)?),
            Self::Hip(buffer) => Ok(buffer.copy_to_slice(dst)?),
        }
    }
}

pub enum GpuModule {
    Cuda(simt_cuda::CudaModule),
    Hip(simt_hip::HipModule),
}

impl GpuModule {
    pub unsafe fn new(data: &[u8]) -> Result<Self> {
        match ScopedGpu::current_api() {
            Some(ComputeApi::Cuda) => Ok(Self::Cuda(simt_cuda::CudaModule::new(data)?)),
            Some(ComputeApi::Hip) => Ok(Self::Hip(simt_hip::HipModule::new(data)?)),
            None => Err(Error::NoCurrentDevice),
        }
    }

    pub fn find(capability: i32, kernels: &[(&str, &[u8])]) -> Result<Self> {
        match ScopedGpu::current_api() {
            Some(ComputeApi::Cuda) => Ok(Self::Cuda(simt_cuda::CudaModule::find(
                capability, kernels,
            )?)),
            Some(ComputeApi::Hip) => Ok(Self::Hip(simt_hip::HipModule::find(capability, kernels)?)),
            None => Err(Error::NoCurrentDevice),
        }
    }
}

pub enum GpuStream {
    Cuda(simt_cuda::CudaStream),
    Hip(simt_hip::HipStream),
}

impl GpuStream {
    pub fn new() -> Result<Self> {
        match ScopedGpu::current_api() {
            Some(ComputeApi::Cuda) => Ok(Self::Cuda(simt_cuda::CudaStream::new()?)),
            Some(ComputeApi::Hip) => Ok(Self::Hip(simt_hip::HipStream::new()?)),
            None => Err(Error::NoCurrentDevice),
        }
    }

    pub fn sync(&self) -> Result<()> {
        match self {
            Self::Cuda(stream) => Ok(stream.sync()?),
            Self::Hip(stream) => Ok(stream.sync()?),
        }
    }
}

#[derive(Default, Clone)]
pub struct LaunchParams<'a> {
    pub blocks: (u32, u32, u32),
    pub threads: (u32, u32, u32),
    pub shared_mem: u32,
    pub stream: Option<&'a GpuStream>,
}

impl<'a> Into<CudaLaunchParams<'a>> for LaunchParams<'a> {
    fn into(self) -> CudaLaunchParams<'a> {
        CudaLaunchParams {
            blocks: self.blocks,
            threads: self.threads,
            shared_mem: self.shared_mem,
            stream: self.stream.map(|stream| match stream {
                GpuStream::Cuda(stream) => stream,
                GpuStream::Hip(_) => panic!("expected cuda stream, got hip stream"),
            }),
        }
    }
}

impl<'a> Into<HipLaunchParams<'a>> for LaunchParams<'a> {
    fn into(self) -> HipLaunchParams<'a> {
        HipLaunchParams {
            blocks: self.blocks,
            threads: self.threads,
            shared_mem: self.shared_mem,
            stream: self.stream.map(|stream| match stream {
                GpuStream::Cuda(_) => panic!("expected hip stream, got cuda stream"),
                GpuStream::Hip(stream) => stream,
            }),
        }
    }
}

pub enum Kernel<T> {
    Cuda(simt_cuda::CudaKernel<T>),
    Hip(simt_hip::HipKernel<T>),
}

impl<T> Kernel<T> {
    pub fn new(module: &GpuModule, name: &str) -> Result<Self> {
        match module {
            GpuModule::Cuda(module) => Ok(Self::Cuda(CudaKernel::new(module, name)?)),
            GpuModule::Hip(module) => Ok(Self::Hip(HipKernel::new(module, name)?)),
        }
    }
}

macro_rules! impl_kernel {
    (($($ty_param:ident),*), ($($ty_idx:tt),*)) => {
        impl<$($ty_param: KernelParam),*> Kernel<($($ty_param),*,)> {
            pub fn launch(
                &self,
                launch_params: LaunchParams,
                params: ($($ty_param),*,),
            ) -> Result<()> {
                match self {
                    Self::Cuda(kernel) => Ok(kernel.launch(launch_params.into(), params)?),
                    Self::Hip(kernel) => Ok(kernel.launch(launch_params.into(), params)?),
                }
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
