#pragma once

extern "C"
{
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include "CPUConverter.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Robust color-accurate converter dynamically handling pixel formats to RGB24.
 */
class AutoToRGBConverter : public ConverterBase
{
  public:
    AutoToRGBConverter()
        : ConverterBase(), sws_ctx(nullptr), last_src_fmt(AV_PIX_FMT_NONE),
          last_dst_fmt(AV_PIX_FMT_NONE), last_src_colorspace(AVCOL_SPC_UNSPECIFIED),
          last_src_color_range(AVCOL_RANGE_UNSPECIFIED), last_width(0), last_height(0)
    {
    }

    ~AutoToRGBConverter() override
    {
        if (sws_ctx)
            sws_freeContext(sws_ctx);
    }

    void convert(celux::Frame& frame, void* buffer) override
    {
        AVFrame* av_frame = frame.get();
        const AVPixelFormat src_fmt = frame.getPixelFormat();
        const int width = frame.getWidth();
        const int height = frame.getHeight();

        // 1) Derive effective bit depth from the frame itself
        const int bit_depth = effective_bit_depth_from_frame(av_frame);

        // 2) Choose destination format/stride accordingly
        const AVPixelFormat dst_fmt =
            (bit_depth <= 8) ? AV_PIX_FMT_RGB24 : AV_PIX_FMT_RGB48LE;
        const int elem_size = (bit_depth <= 8) ? 1 : 2; // bytes per channel
        const int channels = 3;

        // 3) Colorspace defaults
        AVColorSpace src_colorspace = av_frame->colorspace;
        if (src_colorspace == AVCOL_SPC_UNSPECIFIED)
            src_colorspace = (height > 576) ? AVCOL_SPC_BT709 : AVCOL_SPC_BT470BG;

        AVColorRange src_color_range = av_frame->color_range;
        if (src_color_range == AVCOL_RANGE_UNSPECIFIED)
            src_color_range = AVCOL_RANGE_MPEG;

        // 4) (Re)build sws context if anything changed
        if (!sws_ctx || src_fmt != last_src_fmt || dst_fmt != last_dst_fmt ||
            src_colorspace != last_src_colorspace ||
            src_color_range != last_src_color_range || width != last_width ||
            height != last_height)
        {
            if (sws_ctx)
            {
                sws_freeContext(sws_ctx);
                sws_ctx = nullptr;
            }

            sws_ctx = sws_getContext(width, height, src_fmt, width, height, dst_fmt,
                                     SWS_BICUBIC, nullptr, nullptr, nullptr);
            if (!sws_ctx)
                throw std::runtime_error("Failed to initialize swsContext");

            const int* srcCoeffs = sws_getCoefficients(src_colorspace);
            const int* dstCoeffs = sws_getCoefficients(AVCOL_SPC_BT709);
            const int srcRange = (src_color_range == AVCOL_RANGE_JPEG) ? 1 : 0;

            int ok = sws_setColorspaceDetails(sws_ctx, srcCoeffs, srcRange, dstCoeffs,
                                              1, 0, 1 << 16, 1 << 16);
            if (ok < 0)
                CELUX_WARN("sws_setColorspaceDetails returned {}", ok);

            last_src_fmt = src_fmt;
            last_dst_fmt = dst_fmt;
            last_src_colorspace = src_colorspace;
            last_src_color_range = src_color_range;
            last_width = width;
            last_height = height;
        }

        // 5) Do the conversion
        const uint8_t* srcData[4] = {av_frame->data[0], av_frame->data[1],
                                     av_frame->data[2], av_frame->data[3]};
        const int srcLineSize[4] = {av_frame->linesize[0], av_frame->linesize[1],
                                    av_frame->linesize[2], av_frame->linesize[3]};

        uint8_t* dstData[4] = {static_cast<uint8_t*>(buffer), nullptr, nullptr,
                               nullptr};
        const int dstLineSize[4] = {width * channels * elem_size, 0, 0, 0};

        const int result =
            sws_scale(sws_ctx, srcData, srcLineSize, 0, height, dstData, dstLineSize);
        if (result != height)
            throw std::runtime_error("sws_scale failed or incomplete");
    }

  private:
    SwsContext* sws_ctx;
    AVPixelFormat last_src_fmt;
    AVPixelFormat last_dst_fmt;
    AVColorSpace last_src_colorspace;
    AVColorRange last_src_color_range;
    int last_width, last_height;
};

} // namespace cpu
} // namespace conversion
} // namespace celux