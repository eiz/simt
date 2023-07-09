use core::cell::Cell;

use alloc::sync::Arc;
use simt_cuda::{CudaDevice, CudaPhysicalDevice, ScopedCudaDevice};
use simt_hip::{HipDevice, HipPhysicalDevice, ScopedHipDevice};
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
