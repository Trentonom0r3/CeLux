#pragma once

#include "conversion.hpp"
#include "Frame.hpp"
#include <cuda_runtime.h>
extern "C"
{
    /**
     * @brief Converts NV12 format to RGB using CUDA.
     *
     * @param yPlane Pointer to the Y plane of the NV12 data.
     * @param uvPlane Pointer to the UV plane of the NV12 data.
     * @param width Width of the frame.
     * @param height Height of the frame.
     * @param yStride Stride of the Y plane.
     * @param uvStride Stride of the UV plane.
     * @param rgbOutput Pointer to the output RGB data.
     * @param rgbStride Stride of the RGB output.
     * @param stream CUDA stream to be used for conversion.
     */
    void nv12_to_rgb(const unsigned char* yPlane, const unsigned char* uvPlane,
                     int width, int height, int yStride, int uvStride,
                     unsigned char* rgbOutput, int rgbStride, cudaStream_t stream);
}
namespace ffmpy
{
namespace conversion
{
template <typename T> class NV12ToRGB : public ConverterBase<T>
{
  public:
    using ConverterBase<T>::ConverterBase; // Inherit constructors

    void convert(Frame& frame, T* buffer);
};

// Template Definitions

// Convert Method for uint8_t
template <> inline void NV12ToRGB<uint8_t>::convert(Frame& frame, uint8_t* buffer)
{
    const unsigned char* yPlane = frame.getData(0);
    const unsigned char* uvPlane = frame.getData(1);
    int yStride = frame.getLineSize(0);
    int uvStride = frame.getLineSize(1);
    int width = frame.getWidth();
    int height = frame.getHeight();
    int rgbStride = width * 3;

    // Launch CUDA kernel
    nv12_to_rgb(yPlane, uvPlane, width, height, yStride, uvStride, buffer, rgbStride,
                this->conversionStream);
}

// Convert Method for float
template <> inline void NV12ToRGB<float>::convert(Frame& frame, float* buffer)
{
    const unsigned char* yPlane = frame.getData(0);
    const unsigned char* uvPlane = frame.getData(1);
    int yStride = frame.getLineSize(0);
    int uvStride = frame.getLineSize(1);
    int width = frame.getWidth();
    int height = frame.getHeight();
    int rgbStride = width * 3;

    // Conversion logic for float buffer
    // Implement CUDA kernel or other logic as needed
}

// Convert Method for __half
template <> inline void NV12ToRGB<__half>::convert(Frame& frame, __half* buffer)
{
    const unsigned char* yPlane = frame.getData(0);
    const unsigned char* uvPlane = frame.getData(1);
    int yStride = frame.getLineSize(0);
    int uvStride = frame.getLineSize(1);
    int width = frame.getWidth();
    int height = frame.getHeight();
    int rgbStride = width * 3;

    // Conversion logic for __half buffer
    // Implement CUDA kernel or other logic as needed
}
} // namespace conversions
} // namespace ffmpy
