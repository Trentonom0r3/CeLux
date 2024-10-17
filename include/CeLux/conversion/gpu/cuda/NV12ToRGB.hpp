// NV12ToRGB.hpp

#pragma once

#include "BaseConverter.hpp"
#include "Frame.hpp"

extern "C"
{
    // Host functions for different data types
    void nv12_to_rgb(const unsigned char* yPlane, const unsigned char* uvPlane,
                     int width, int height, int yStride, int uvStride,
                     unsigned char* rgbOutput, int rgbStride, cudaStream_t stream);

    void nv12_to_rgb_float(const unsigned char* yPlane, const unsigned char* uvPlane,
                           int width, int height, int yStride, int uvStride,
                           float* rgbOutput, int rgbStride, cudaStream_t stream);

    void nv12_to_rgb_half(const unsigned char* yPlane, const unsigned char* uvPlane,
                          int width, int height, int yStride, int uvStride,
                          __half* rgbOutput, int rgbStride, cudaStream_t stream);
}
namespace celux
{
namespace conversion
{
namespace gpu
{
namespace cuda
{

template <typename T> class NV12ToRGB : public ConverterBase<T>
{
  public:
    NV12ToRGB();
    NV12ToRGB(cudaStream_t stream);
    ~NV12ToRGB();

    void convert(celux::Frame& frame, void* buffer) override;
};

// Template Definitions

template <typename T> NV12ToRGB<T>::NV12ToRGB() : ConverterBase<T>()
{
}

template <typename T>
NV12ToRGB<T>::NV12ToRGB(cudaStream_t stream) : ConverterBase<T>(stream)
{
}

template <typename T> NV12ToRGB<T>::~NV12ToRGB()
{
}

template <typename T> void NV12ToRGB<T>::convert(celux::Frame& frame, void* buffer)
{
    const unsigned char* yPlane = frame.getData(0);
    const unsigned char* uvPlane = frame.getData(1);
    int yStride = frame.getLineSize(0);
    int uvStride = frame.getLineSize(1);
    int width = frame.getWidth();
    int height = frame.getHeight();
    int rgbStride = width * 3;

    if constexpr (std::is_same<T, uint8_t>::value)
    {
        // Call the kernel for uint8_t
        nv12_to_rgb(yPlane, uvPlane, width, height, yStride, uvStride,
                    static_cast<uint8_t*>(buffer), rgbStride, this->conversionStream);
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        // Call the kernel for float
        nv12_to_rgb_float(yPlane, uvPlane, width, height, yStride, uvStride,
                          static_cast<float*>(buffer), rgbStride,
                          this->conversionStream);
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        // Call the kernel for __half
        nv12_to_rgb_half(yPlane, uvPlane, width, height, yStride, uvStride,
                         static_cast<__half*>(buffer), rgbStride,
                         this->conversionStream);
    }
    else
    {
        static_assert(sizeof(T) == 0, "Unsupported data type for NV12ToRGB");
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
