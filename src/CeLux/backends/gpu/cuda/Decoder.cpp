// CUDA Decoder.cpp
#include "backends/gpu/cuda/Decoder.hpp"
// include cuda streams
using namespace celux::error;

namespace celux::backends::gpu::cuda
{
void Decoder::initHWAccel()
{
    CELUX_DEBUG("GPU DECODER: Initializing hardware acceleration for CUDA Decoder");

    // Find the hardware device type
    CELUX_DEBUG("GPU DECODER: Searching for HW device type 'cuda'");
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
    if (type == AV_HWDEVICE_TYPE_NONE)
    {
        CELUX_ERROR("Failed to find HW device type: cuda");
        throw CxException("Failed to find HW device type: cuda");
    }

    CELUX_DEBUG("GPU DECODER: HW device type 'cuda' found: {}", av_hwdevice_get_type_name(type));

    // Initialize hardware device context
    CELUX_DEBUG("GPU DECODER: Creating HW device context for type: {}",
                av_hwdevice_get_type_name(type));

    AVBufferRef* hw_ctx = nullptr;
    int ret = av_hwdevice_ctx_create(&hw_ctx, type, nullptr, nullptr, 0);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to create HW device context for 'cuda': error code {}",
                    ret);
        FF_CHECK_MSG(ret, std::string("Failed to create HW device context:"));
    }
    hwDeviceCtx.reset(hw_ctx);
    CELUX_DEBUG("GPU DECODER: HW device context for 'cuda' initialized successfully");
}

} // namespace celux::backends::gpu::cuda
