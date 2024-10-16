// CUDA Decoder.cpp
#include "backends/gpu/cuda/Decoder.hpp"
using namespace ffmpy::error;

namespace ffmpy::backends::gpu::cuda
{
void Decoder::initCodecContext(const AVCodec* codec)
{
    // Call base class implementation
    ffmpy::Decoder::initCodecContext(codec);

    // Set hardware-specific get_format
    if (hwDeviceCtx)
    {
        codecCtx->get_format = Decoder::getHWFormat; // Assign the static function
    }
}

void Decoder::initHWAccel()
{
    // Find the hardware device type
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
    if (type == AV_HWDEVICE_TYPE_NONE)
    {
        throw FFException("Failed to find HW device type: cuda");
    }

    // Initialize hardware device context
    AVBufferRef* hw_ctx = nullptr;
    FF_CHECK_MSG(av_hwdevice_ctx_create(&hw_ctx, type, nullptr, nullptr, 0),
                 std::string("Failed to create HW device context:"));
    hwDeviceCtx.reset(hw_ctx);
}

enum AVPixelFormat Decoder::getHWFormat(AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts)
{
    for (const enum AVPixelFormat* p = pix_fmts; *p != -1; p++)
    {
        if (*p == AV_PIX_FMT_CUDA)
        {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}

Decoder::~Decoder()
{
}
} // namespace ffmpy::backends::gpu::cuda
