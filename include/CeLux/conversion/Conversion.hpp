#pragma once
#ifndef CONVERSION_HPP
#define CONVERSION_HPP

#ifdef CUDA_ENABLED
#include <CPUConverters.hpp>
#include <GPUConverters.hpp>
#else
#include <CPUConverters.hpp>
#endif
namespace celux
{
enum class ConversionType
{
    RGBToNV12,
    NV12ToRGB,
    YUV420ToRGB,
    RGBToYUV420,
};

enum class SupportedFormats
{
    YUV420P,
    NV12,
    YUV420P10LE,
    P010LE,
    RGB24,
    RGB48
};

enum class EncodingFormats
{
    YUV420P,     // 8 bit sw format -- requires input to be in Uint8, rgb24
                 // NV12, // 8 bit hw format
    YUV420P10LE, // 10 bit sw format -- requires input to be in Uint16, rgb48
    P010LE,      // 10 bit hw format
};

static AVPixelFormat getEncoderPixelFormat(EncodingFormats format)
{
    switch (format)
    {
    case EncodingFormats::YUV420P:
        return AV_PIX_FMT_YUV420P;
    case EncodingFormats::YUV420P10LE:
        return AV_PIX_FMT_YUV420P10LE;
    case EncodingFormats::P010LE:
		return AV_PIX_FMT_P010LE;
    default:
        return AV_PIX_FMT_YUV420P;
    }
}

static AVPixelFormat getConverterPixelFormat(EncodingFormats format)
{
    switch (format)
    {
    case EncodingFormats::YUV420P:
        return AV_PIX_FMT_RGB24;
    case EncodingFormats::YUV420P10LE:
        return AV_PIX_FMT_RGB48LE;
    case EncodingFormats::P010LE:
        return AV_PIX_FMT_RGB48LE;
    default:
        return AV_PIX_FMT_RGB24;
    }
}

static int getEncoderDepth(EncodingFormats format)
{
    switch (format)
    {
    case EncodingFormats::YUV420P:
        return 8;
    case EncodingFormats::YUV420P10LE:
    case EncodingFormats::P010LE:
        return 10;
    default:
        return 8;
    }
};

static std::string encoderFormatToString(EncodingFormats format)
{
    switch (format)
    {
    case EncodingFormats::YUV420P:
        return "YUV420P";
    case EncodingFormats::YUV420P10LE:
        return "YUV420P10LE";
    case EncodingFormats::P010LE:
		return "P010LE";
    default:
        return "YUV420P";
    }
}

static int getEncoderDepth(AVPixelFormat format)
{
    switch (format)
    {
    case AV_PIX_FMT_YUV420P:
        return 8;
    case AV_PIX_FMT_YUV420P10LE:
    case AV_PIX_FMT_P010LE:
        return 10;
    default:
        return 8;
    }
};

enum SupportedBitDepth
{
    DEPTH8 = 8,
    DEPTH10 = 10,
};

enum class backend
{
    CPU,
    CUDA
};

enum class SupportedCodecs
{
    H264,
    H265,
    H264_CUDA, // AKA H264 NVENC
    H265_CUDA, // AKA HEVC NVENC
};

static std::string codecToString(SupportedCodecs codec)
{
    switch (codec)
    {
    case SupportedCodecs::H264:
        return "libx264";
    case SupportedCodecs::H265:
        return "libx265";
    case SupportedCodecs::H264_CUDA:
        return "h264_nvenc";
    case SupportedCodecs::H265_CUDA:
        return "hevc_nvenc";
    }
}


} // namespace celux

#endif
