// BGRToNV12.hpp

#pragma once

#include "Frame.hpp"
#include "BaseConverter.hpp"
#include <cuda_runtime.h>
#include <type_traits>

extern "C"
{
                     
    void bgr_to_nv12(const unsigned char* bgrInput, int width, int height,
                     int bgrStride, unsigned char* yPlane, unsigned char* uvPlane,
                     int yStride, int uvStride, cudaStream_t stream = 0);
    void bgr_to_nv12_float(const float* bgrInput, int width, int height, int bgrStride,
                           unsigned char* yPlane, unsigned char* uvPlane, int yStride,
                           int uvStride, cudaStream_t stream = 0);
    void bgr_to_nv12_half(const __half* bgrInput, int width, int height, int bgrStride,
                          unsigned char* yPlane, unsigned char* uvPlane, int yStride,
                          int uvStride, cudaStream_t stream = 0);
}

namespace celux
{
namespace conversion
{
namespace gpu
{
namespace cuda
{

template <typename T> class BGRToNV12 : public ConverterBase<T>
{
  public:
    BGRToNV12();
    BGRToNV12(cudaStream_t stream);
    ~BGRToNV12();

    void convert(celux::Frame& frame, void* buffer) override;
};

// Template Definitions

template <typename T> BGRToNV12<T>::BGRToNV12() : ConverterBase<T>()
{
}

template <typename T>
BGRToNV12<T>::BGRToNV12(cudaStream_t stream) : ConverterBase<T>(stream)
{
}

template <typename T> BGRToNV12<T>::~BGRToNV12()
{
}

template <typename T> void BGRToNV12<T>::convert(celux::Frame& frame, void* buffer)
{
    const unsigned char* bgrInput = static_cast<const unsigned char*>(buffer);
    unsigned char* yPlane = frame.getData(0);
    unsigned char* uvPlane = frame.getData(1);
    int yStride = frame.getLineSize(0);
    int uvStride = frame.getLineSize(1);
    int width = frame.getWidth();
    int height = frame.getHeight();
    int bgrStride = width * 3;

    if constexpr (std::is_same<T, uint8_t>::value)
    {
        // Call the kernel for uint8_t
        bgr_to_nv12(bgrInput, width, height, bgrStride, yPlane, uvPlane, yStride,
                    uvStride, this->conversionStream);
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        // Call the kernel for float
        bgr_to_nv12_float(reinterpret_cast<const float*>(bgrInput), width, height,
                          bgrStride, yPlane, uvPlane, yStride, uvStride,
                          this->conversionStream);
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        // Call the kernel for __half
        bgr_to_nv12_half(reinterpret_cast<const __half*>(bgrInput), width, height,
                         bgrStride, yPlane, uvPlane, yStride, uvStride,
                         this->conversionStream);
    }
    else
    {
        static_assert(sizeof(T) == 0, "Unsupported data type for BGRToNV12");
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

} // namespace cuda
} // namespace gpu
} // namespace conversion
} // namespace celux
