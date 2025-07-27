#pragma once
#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <Decoders.hpp>

using ConverterKey = std::tuple<bool, AVPixelFormat>;

// Hash function for ConverterKey
struct ConverterKeyHash
{
    std::size_t operator()(const std::tuple<bool, AVPixelFormat>& key) const
    {
        return std::hash<bool>()(std::get<0>(key)) ^
               std::hash<int>()(static_cast<int>(std::get<1>(key)));
    }
};

namespace celux
{

/**
 * @brief Factory class to create Decoders, Encoders, and Converters based on backend
 * and configuration.
 */
class Factory
{
  public:
    /**
     * @brief Creates a Decoder instance based on the specified backend.
     *
     * @param backend Backend type (CPU or CUDA).
     * @param filename Path to the video file.
     * @param converter Unique pointer to the IConverter instance.
     * @return std::unique_ptr<Decoder> Pointer to the created Decoder.
     */
    static std::shared_ptr<Decoder>
    createDecoder(torch::Device device, const std::string& filename, int numThreads
                  )
    {

        return std::make_shared<celux::backends::cpu::Decoder>(filename, numThreads
                                                               );
    }

   

  private:
    // Helper function to infer bit depth from AVPixelFormat
    static int inferBitDepth(AVPixelFormat pixfmt)
    {
        switch (pixfmt)
        {
        // Already existing 8-bit formats
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_BGR24:
        case AV_PIX_FMT_RGBA:
        case AV_PIX_FMT_BGRA:
        case AV_PIX_FMT_GBRP:
            return 8;

        // Already existing 10-bit formats
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_YUV422P10LE:
        case AV_PIX_FMT_P010LE:
        case AV_PIX_FMT_RGB48LE:
            return 10;

        // ------------------------
        // NEW ENTRIES (one by one)
        // ------------------------
        // 8-bit planar 4:2:2
        case AV_PIX_FMT_YUV422P:
            return 8;

        // 8-bit planar 4:4:4
        case AV_PIX_FMT_YUV444P:
            return 8;

        // 10-bit planar 4:4:4
        case AV_PIX_FMT_YUV444P10LE:
            return 10;

        // 12-bit planar 4:2:0
        case AV_PIX_FMT_YUV420P12LE:
            return 12;

        // 12-bit planar 4:2:2
        case AV_PIX_FMT_YUV422P12LE:
            return 12;

        // 12-bit planar 4:4:4
        case AV_PIX_FMT_YUV444P12LE:
            return 12;

        // ProRes4444 often decodes to YUVA444P10 or YUVA444P16.
        case AV_PIX_FMT_YUVA444P10LE:
            return 10;
        // Or if you also want to handle 12-bit alpha:
        case AV_PIX_FMT_YUVA444P12LE:
            return 12;

        default:
            throw std::invalid_argument(
                std::string("Unknown pixel format for bit depth inference: ") +
                av_get_pix_fmt_name(pixfmt));
        }
    }
};

} // namespace celux

#endif // FACTORY_HPP
