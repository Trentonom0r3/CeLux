// FFCore.hpp
#ifndef FFMPEG_WRAPPER_HPP
#define FFMPEG_WRAPPER_HPP

#include <algorithm>
#include <cstdint> // For fixed-width integer types
#include <exception>
#include <iostream>
#include <memory>
#include <ostream>   // For std::ostream
#include <stdexcept> // For std::runtime_error
#include <string>
#include <thread>
#include <Logger.hpp>
#include <torch/extension.h>
#include <vector>
#include <optional>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h> // For handling channel layout information
#include <libavutil/dict.h>
#include <libavutil/error.h>          // For error codes
#include <libavutil/imgutils.h>       // For image utilities
#include <libavutil/opt.h>            // For AVOptions
#include <libavutil/pixfmt.h>         // For pixel formats
#include <libavutil/samplefmt.h>      // For handling sample format information
#include <libswresample/swresample.h> // Include for SwrContext and resampling functions
                                      // hwaccel
#include <libavutil/hwcontext.h>
    // audio headers
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/audio_fifo.h>
#include <libswscale/swscale.h>

}

namespace celux
{
/**
 * @brief Utility function to get a suitable hardware configuration for a codec.
 *
 * @param codec A pointer to the codec for which hardware configuration is needed.
 * @return A pointer to the suitable AVCodecHWConfig, or nullptr if none found.
 */
inline const AVCodecHWConfig* getSuitableHWConfig(const AVCodec* codec)
{
    const AVCodecHWConfig* hwConfig = nullptr;
    int index = 0;
    while ((hwConfig = avcodec_get_hw_config(codec, index++)))
    {
        // Check if the configuration supports hardware device context
        if (hwConfig->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)
        {
            return hwConfig;
        }
    }
    return nullptr; // No suitable hardware configuration found
}

/**
 * @brief Utility function to log supported hardware configurations for a codec.
 *
 * @param codec A pointer to the codec for which hardware configurations are to be
 * logged.
 */
inline void logSupportedHardwareConfigs(const AVCodec* codec)
{
    const AVCodecHWConfig* hwConfig = nullptr;
    int index = 0;
    while ((hwConfig = avcodec_get_hw_config(codec, index++)))
    {
        // Get the device type name for the hardware configuration
        const char* deviceTypeName = av_hwdevice_get_type_name(hwConfig->device_type);
        if (deviceTypeName)
        {
            std::cout << "Supported hardware config: " << deviceTypeName << std::endl;
        }
    }
}

/**
 * @brief Utility function to convert FFmpeg error codes to readable strings.
 *
 * @param errorCode The FFmpeg error code.
 * @return A string representation of the error.
 */
inline std::string errorToString(int errorCode)
{
    char errBuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errorCode, errBuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(errBuf);
}

/**
 * @brief Checks if hardware acceleration is supported for a given codec.
 *
 * @param codec A pointer to the codec to check for hardware acceleration support.
 * @return True if hardware acceleration is supported, false otherwise.
 */
inline bool isHardwareAccelerationSupported(const AVCodec* codec)
{
    // Iterate over all hardware configurations for the codec
    for (int i = 0;; i++)
    {
        const AVCodecHWConfig* hwConfig = avcodec_get_hw_config(codec, i);
        if (!hwConfig)
        {
            break; // No more configurations
        }

        // Check if the codec has any hardware acceleration capabilities
        if (hwConfig->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)
        {
            return true;
        }
    }
    return false;
}

} // namespace celux

#endif // FFMPEG_WRAPPER_HPP