// FFCore.hpp
#ifndef FFMPEG_WRAPPER_HPP
#define FFMPEG_WRAPPER_HPP

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/channel_layout.h>  // For handling channel layout information
#include <libavutil/samplefmt.h>        // For handling sample format information
#include <libswresample/swresample.h>  // Include for SwrContext and resampling functions
}

#include <string>
#include <memory>

namespace FFmpeg {

    // Utility function to convert FFmpeg error codes to readable strings
    inline std::string errorToString(int errorCode) {
        char errBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(errorCode, errBuf, AV_ERROR_MAX_STRING_SIZE);
        return std::string(errBuf);
    }

} // namespace FFmpeg

#endif // FFMPEG_WRAPPER_HPP
