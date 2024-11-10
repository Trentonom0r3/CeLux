#pragma once

#include "CPUConverter.hpp"

namespace celux
{
namespace conversion
{
namespace cpu
{

class RGBToRGB : public ConverterBase
{
  public:
    RGBToRGB() : ConverterBase()
    {
    }

    ~RGBToRGB()
    {
    }

    void convert(celux::Frame& frame, void* buffer) override
    {
        // Verify the pixel format
        if (frame.getPixelFormat() != AV_PIX_FMT_RGB24)
        {
            throw std::runtime_error("Input frame is not in RGB24 format");
        }

        int width = frame.getWidth();
        int height = frame.getHeight();
        int bytesPerPixel = 3; // For AV_PIX_FMT_RGB24

        // Calculate the size of the tightly packed image
        int numBytes = width * height * bytesPerPixel;

        // Use av_image_copy_to_buffer to copy the data efficiently
        int ret = av_image_copy_to_buffer(static_cast<uint8_t*>(buffer), numBytes,
                                          frame.get()->data, frame.get()->linesize,
                                          AV_PIX_FMT_RGB24, width, height, 1);
        if (ret < 0)
        {
            throw std::runtime_error("av_image_copy_to_buffer failed");
        }
    }
};

} // namespace cpu
} // namespace conversion
} // namespace celux
