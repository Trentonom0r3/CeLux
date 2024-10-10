// RGBToNV12.hpp

#pragma once

#include "conversion.hpp"
#include "Frame.hpp"
#include <cuda_runtime.h>
#include <type_traits>

extern "C"
{

}
namespace ffmpy
{
namespace conversion
{

template <typename T> class RGBToNV12 : public ConverterBase<T>
{
  public:
    RGBToNV12();
    RGBToNV12(cudaStream_t stream);
    ~RGBToNV12();

    void convert(ffmpy::Frame& frame, void* buffer) override;
};

// Template Definitions

template <typename T> RGBToNV12<T>::RGBToNV12() : ConverterBase<T>()
{
}

template <typename T>
RGBToNV12<T>::RGBToNV12(cudaStream_t stream) : ConverterBase<T>(stream)
{
}

template <typename T> RGBToNV12<T>::~RGBToNV12()
{
}

template <typename T> void RGBToNV12<T>::convert(ffmpy::Frame& frame, void* buffer)
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
       // nv12_to_rgb(yPlane, uvPlane, width, height, yStride, uvStride,
           //         static_cast<uint8_t*>(buffer), rgbStride, this->conversionStream);
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        // Call the kernel for float
      //  nv12_to_rgb_float(yPlane, uvPlane, width, height, yStride, uvStride,
              //            static_cast<float*>(buffer), rgbStride,
               //           this->conversionStream);
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        // Call the kernel for __half
      //  nv12_to_rgb_half(yPlane, uvPlane, width, height, yStride, uvStride,
        //                 static_cast<__half*>(buffer), rgbStride,
       //                  this->conversionStream);
    }
    else
    {
        static_assert(sizeof(T) == 0, "Unsupported data type for RGBToNV12");
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
