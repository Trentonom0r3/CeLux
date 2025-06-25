#pragma once

extern "C"
{
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include "CPUConverter.hpp" // Assuming this defines ConverterBase
#include <iostream>
#include <stdexcept>

namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief General-purpose converter that dynamically handles
 *        any pixel format → RGB24 on CPU using swscale.
 */
class AutoToRGB24Converter : public ConverterBase
{
  public:
    AutoToRGB24Converter() : ConverterBase(), sws_ctx(nullptr)
    {
        CELUX_DEBUG("Initializing AutoToRGB24Converter");
    }

    ~AutoToRGB24Converter() override
    {
        if (sws_ctx)
        {
            sws_freeContext(sws_ctx);
            sws_ctx = nullptr;
        }
    }

    /**
     * @brief Converts any supported pixel format to RGB24, writing directly into
     * provided buffer.
     *
     * @param frame celux::Frame object containing AVFrame data
     * @param buffer Preallocated RGB24 tensor buffer (HWC layout, 3 channels)
     */
    void convert(celux::Frame& frame, void* buffer) override
    {
        const AVPixelFormat src_fmt = frame.getPixelFormat();

        if (!sws_ctx)
        {
            sws_ctx =
                sws_getContext(frame.getWidth(), frame.getHeight(), src_fmt,
                               frame.getWidth(), frame.getHeight(), AV_PIX_FMT_RGB24,
                               SWS_BILINEAR, nullptr, nullptr, nullptr);

            if (!sws_ctx)
            {
                throw std::runtime_error(
                    "Failed to initialize swsContext for AutoToRGB24Converter");
            }

            sws_setColorspaceDetails(sws_ctx, sws_getCoefficients(SWS_CS_DEFAULT), 0,
                                     sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16,
                                     1 << 16);
        }

        const uint8_t* srcData[4] = {frame.getData(0), frame.getData(1),
                                     frame.getData(2), frame.getData(3)};

        int srcLineSize[4] = {frame.getLineSize(0), frame.getLineSize(1),
                              frame.getLineSize(2), frame.getLineSize(3)};

        uint8_t* dstData[4] = {static_cast<uint8_t*>(buffer), nullptr, nullptr,
                               nullptr};
        int dstLineSize[4] = {frame.getWidth() * 3, 0, 0, 0}; // RGB24 = 3 bytes/pixel

        int result = sws_scale(sws_ctx, srcData, srcLineSize, 0, frame.getHeight(),
                               dstData, dstLineSize);

        if (result <= 0)
        {
            throw std::runtime_error("sws_scale failed in AutoToRGB24Converter");
        }
    }

  private:
    SwsContext* sws_ctx;
};

} // namespace cpu
} // namespace conversion
} // namespace celux
