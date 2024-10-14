// NV12ToBGR.hpp

#pragma once

#include "conversion.hpp"
#include "Frame.hpp"
#include <cuda_runtime.h>
#include <type_traits>

extern "C"
{
    // Host functions for different data types
    void nv12_to_bgr(const unsigned char* yPlane, const unsigned char* uvPlane,
                     int width, int height, int yStride, int uvStride,
                     unsigned char* bgrOutput, int bgrStride, cudaStream_t stream);

    void nv12_to_bgr_float(const unsigned char* yPlane, const unsigned char* uvPlane,
                           int width, int height, int yStride, int uvStride,
                           float* bgrOutput, int bgrStride, cudaStream_t stream);

    void nv12_to_bgr_half(const unsigned char* yPlane, const unsigned char* uvPlane,
                          int width, int height, int yStride, int uvStride,
                          __half* bgrOutput, int bgrStride, cudaStream_t stream);
}
namespace ffmpy
{
namespace conversion
{

template <typename T> class NV12ToBGR : public ConverterBase<T>
{
  public:
    NV12ToBGR();
    NV12ToBGR(cudaStream_t stream);
    ~NV12ToBGR();

    void convert(ffmpy::Frame& frame, void* buffer) override;
};

// Template Definitions

template <typename T> NV12ToBGR<T>::NV12ToBGR() : ConverterBase<T>()
{
}

template <typename T>
NV12ToBGR<T>::NV12ToBGR(cudaStream_t stream) : ConverterBase<T>(stream)
{
}

template <typename T> NV12ToBGR<T>::~NV12ToBGR()
{
}

template <typename T> void NV12ToBGR<T>::convert(ffmpy::Frame& frame, void* buffer)
{
    const unsigned char* yPlane = frame.getData(0);
    const unsigned char* uvPlane = frame.getData(1);
    int yStride = frame.getLineSize(0);
    int uvStride = frame.getLineSize(1);
    int width = frame.getWidth();
    int height = frame.getHeight();
    int bgrStride = width * 3;

    if constexpr (std::is_same<T, uint8_t>::value)
    {
        // Call the kernel for uint8_t
        nv12_to_bgr(yPlane, uvPlane, width, height, yStride, uvStride,
                    static_cast<uint8_t*>(buffer), bgrStride, this->conversionStream);
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        // Call the kernel for float
        nv12_to_bgr_float(yPlane, uvPlane, width, height, yStride, uvStride,
                          static_cast<float*>(buffer), bgrStride,
                          this->conversionStream);
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        // Call the kernel for __half
        nv12_to_bgr_half(yPlane, uvPlane, width, height, yStride, uvStride,
                         static_cast<__half*>(buffer), bgrStride,
                         this->conversionStream);
    }
    else
    {
        static_assert(sizeof(T) == 0, "Unsupported data type for NV12ToBGR");
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

} // namespace conversion
} // namespace ffmpy
