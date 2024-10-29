#pragma once
#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <Decoders.hpp>
#include <torch/extension.h>

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
#ifdef CUDA_ENABLED
static cudaStream_t checkStream(std::optional<torch::Stream> stream)
{
    if (stream.has_value())
    {
        c10::cuda::CUDAStream cuda_stream(stream.value());
        return cuda_stream.stream();
    }
    else
    {
        return c10::cuda::getStreamFromPool();
    }
}
#endif

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
    static std::unique_ptr<Decoder> createDecoder(torch::Device device,
                                                  const std::string& filename, int numThreads,
                                                  std::optional<torch::Stream> stream)
    {
        if (device.is_cpu())
        {
            return std::make_unique<celux::backends::cpu::Decoder>(filename, std::nullopt, numThreads);
        }
#ifdef CUDA_ENABLED
        else if (device.is_cuda())
        {
            return std::make_unique<celux::backends::gpu::cuda::Decoder>(filename, stream, numThreads);
        }
#endif // CUDA_ENABLED
        else
        {
            throw std::invalid_argument("Unsupported backend: " + device.str());
        }
    }

    /**
     * @brief Creates a Converter instance based on the specified backend and pixel
     * format.
     *
     * @param device Device type (CPU or CUDA).
     * @param pixfmt Pixel format.
     * @param stream Optional stream for CUDA operations.
     * @return std::unique_ptr<celux::conversion::IConverter> Pointer to the created
     * Converter.
     */
    static std::unique_ptr<celux::conversion::IConverter>
    createConverter(const torch::Device& device, AVPixelFormat pixfmt,
                    const std::optional<torch::Stream>& stream = std::nullopt)
    {
        using namespace celux::conversion; // For IConverter

        // Determine device type
        bool is_cpu = device.is_cpu();
        bool is_cuda = device.is_cuda();

        // Infer bit depth from pixel format
        int bit_depth = inferBitDepth(pixfmt);

        // Define the key based on device and pixel format
        ConverterKey key = std::make_tuple(is_cpu, pixfmt);

        // Define the factory map
        static const std::unordered_map<ConverterKey,
                                        std::function<std::unique_ptr<IConverter>(
                                            const std::optional<torch::Stream>&)>,
                                        ConverterKeyHash>
            converterMap = {
                // CPU converters
                {std::make_tuple(true, AV_PIX_FMT_YUV420P),
                 [](const std::optional<torch::Stream>&) -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating YUV420PToRGB converter");
                     return std::make_unique<cpu::YUV420PToRGB>();
                 }},

                {std::make_tuple(true, AV_PIX_FMT_YUV420P10LE),
                 [](const std::optional<torch::Stream>&) -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating YUV420P10LEToRGB48 converter");
                     return std::make_unique<cpu::YUV420P10ToRGB48>();
                 }},

                // bgr24
                {std::make_tuple(true, AV_PIX_FMT_BGR24),
                 [](const std::optional<torch::Stream>&) -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating BGR24ToRGB converter");
                     return std::make_unique<cpu::BGRToRGB>();
                 }},
                //rgb24
                {std::make_tuple(true, AV_PIX_FMT_RGB24),
                         [](const std::optional<torch::Stream>&) -> std::unique_ptr<IConverter>
                         {
					 CELUX_DEBUG("Creating RGB24ToRGB converter");
					 return std::make_unique<cpu::RGBToRGB>();
				 }},

#ifdef CUDA_ENABLED
                // CUDA converters
                {std::make_tuple(false, AV_PIX_FMT_NV12),
                 [](const std::optional<torch::Stream>& stream)
                     -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating NV12ToRGB converter");
                     return std::make_unique<gpu::cuda::NV12ToRGB>(checkStream(stream));
                 }},

                {std::make_tuple(false, AV_PIX_FMT_P010LE),
                 [](const std::optional<torch::Stream>& stream)
                     -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating P010LEToRGB48LE converter");
                     // Uncomment and implement the converter when ready
                     return std::make_unique<gpu::cuda::P010LEToRGB>(
                         checkStream(stream));
                 }},

#endif
            };

        // Search for the converter in the map
        auto it = converterMap.find(key);
        if (it != converterMap.end())
        {
            return it->second(stream);
        }

        // If not found, throw an exception with detailed information
        std::string deviceType = is_cpu ? "CPU" : (is_cuda ? "CUDA" : "Unknown");
        throw std::invalid_argument("Unsupported combination - Device: " + deviceType +
                                    ", Bit Depth: " + std::to_string(bit_depth) +
                                    ", Pixel Format: " + av_get_pix_fmt_name(pixfmt));
    }

  private:
    // Helper function to infer bit depth from AVPixelFormat
    static int inferBitDepth(AVPixelFormat pixfmt)
    {
        switch (pixfmt)
        {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_BGR24:
            return 8;
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_P010LE:
        case AV_PIX_FMT_RGB48LE:
            return 10;
        // Add more cases as needed
        default:
            throw std::invalid_argument(
                std::string("Unknown pixel format for bit depth inference: ") +
                av_get_pix_fmt_name(pixfmt));
        }
    }
};

} // namespace celux

#endif // FACTORY_HPP
