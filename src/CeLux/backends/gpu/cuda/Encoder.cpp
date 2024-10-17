#include "backends/gpu/cuda/Encoder.hpp"
using namespace celux::error;
namespace celux::backends::gpu::cuda
{
void Encoder::initHWAccel()
{
    std::cout << "Initializing HW Acceleration\n" << std::endl;
    // Find the hardware device type
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
    if (type == AV_HWDEVICE_TYPE_NONE)
    {
        throw CxException("Failed to find HW device type: CUDA");
    }

    // Initialize hardware device context
    AVBufferRef* hw_ctx = nullptr;
    FF_CHECK(av_hwdevice_ctx_create(&hw_ctx, type, nullptr, nullptr, 0));

    hwDeviceCtx.reset(hw_ctx);

    // Initialize hardware frames context
    AVBufferRef* frames_ctx = av_hwframe_ctx_alloc(hwDeviceCtx.get());
    if (!frames_ctx)
    {
        throw CxException("Failed to create hardware frames context");
    }

    AVHWFramesContext* frames = reinterpret_cast<AVHWFramesContext*>(frames_ctx->data);
    frames->format = codecCtx->pix_fmt;  // Hardware pixel format
    frames->sw_format = AV_PIX_FMT_NV12; // Software pixel format (input format)
    frames->width = properties.width;
    frames->height = properties.height;
    frames->initial_pool_size = 20;

    int ret = av_hwframe_ctx_init(frames_ctx);
    if (ret < 0)
    {
        throw CxException(ret);
    }

    hwFramesCtx.reset(frames_ctx);
}

void Encoder::initCodecContext(const AVCodec* codec, const VideoProperties& props)
{
    // Call base class implementation
    celux::Encoder::initCodecContext(codec, props);

    // If hardware acceleration is enabled, set hw_device_ctx and get_format callback
    if (hwDeviceCtx)
    {
        codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx.get());
        if (!codecCtx->hw_device_ctx)
        {
            throw CxException("Failed to reference HW device context");
        }
//        codecCtx->get_format = getHWFormat;
        codecCtx->pix_fmt = AV_PIX_FMT_CUDA;
    }

    codecCtx->hw_frames_ctx = av_buffer_ref(hwFramesCtx.get());

    // Set encoder options (these can be adjusted as needed)
    av_opt_set(codecCtx->priv_data, "preset", "p5", 0);  // Example: NVENC preset
    av_opt_set(codecCtx->priv_data, "rc", "constqp", 0); // Rate control mode
    av_opt_set(codecCtx->priv_data, "qp", "23", 0);      // Quantization parameter

    // Convert the raw buffer to AVFrame using the converter
    frame.get()->hw_frames_ctx = av_buffer_ref(hwFramesCtx.get());
    frame.get()->format = AV_PIX_FMT_CUDA;
    FF_CHECK(av_hwframe_get_buffer(hwFramesCtx.get(), frame.get(), 0));

    // Open the codec
    FF_CHECK(avcodec_open2(codecCtx.get(), codec, nullptr));
}

enum AVPixelFormat Encoder::getHWFormat(AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts)
{
    for (const enum AVPixelFormat* p = pix_fmts; *p != -1; p++)
    {
        if (*p == AV_PIX_FMT_CUDA) // Example: CUDA pixel format
        {
            return *p;
        }
        // Add additional hardware pixel formats if needed
    }

    return AV_PIX_FMT_NONE;
}
} // namespace celux::backends::gpu::cuda
