#pragma once
#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <Decoders.hpp>
#include <Encoders.hpp> // Assuming you have header files that declare Encoder classes
#include <torch/extension.h>
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
                                                  const std::string& filename,
                                                  std::optional<torch::Stream> stream)
    {
        if (device.is_cpu())
        {
            return std::make_unique<celux::backends::cpu::Decoder>(filename,
                                                                   std::nullopt);
        }
#ifdef CUDA_ENABLED
        else if (device.is_cuda())
        {
            return std::make_unique<celux::backends::gpu::cuda::Decoder>(filename,
                                                                         stream);
        }
#endif // CUDA_ENABLED
        else
        {
            throw std::invalid_argument("Unsupported backend: " + device.str());
        }
    }

    /**
     * @brief Creates an Encoder instance based on the specified backend.
     *
     * @param backend Backend type (CPU or CUDA).
     * @param filename Path to the output video file.
     * @param props Video properties for the encoder (e.g., width, height, fps).
     * @param converter Unique pointer to the IConverter instance.
     * @return std::unique_ptr<Encoder> Pointer to the created Encoder.
     */
    static std::unique_ptr<Encoder>
    createEncoder(torch::Device device, const std::string& filename,
                  const Encoder::VideoProperties& props,
                  std::unique_ptr<celux::conversion::IConverter> converter)
    {
        if (device.is_cpu())
        {
            return std::make_unique<celux::backends::cpu::Encoder>(
                filename, props, std::move(converter));
        }
#ifdef CUDA_ENABLED
        else if (device.is_cuda())
        {
            return std::make_unique<celux::backends::gpu::cuda::Encoder>(
                filename, props, std::move(converter));
        }
#endif
        else
        {
            throw std::invalid_argument("Unsupported backend: " + device.str());
        }
    }

// Type alias for the key used in the converter map
    using ConverterKey = std::tuple<bool, int, AVPixelFormat>;

    // Hash function for the ConverterKey to be used in unordered_map
    struct ConverterKeyHash
    {
        std::size_t operator()(const ConverterKey& key) const
        {
            return std::hash<bool>()(std::get<0>(key)) ^
                   std::hash<int>()(std::get<1>(key)) ^
                   std::hash<int>()(static_cast<int>(std::get<2>(key)));
        }
    };

    // Factory function to retrieve the appropriate converter creator
    static std::unique_ptr<celux::conversion::IConverter>
    createConverter(const torch::Device& device, int bit_depth, AVPixelFormat pixfmt,
                    const std::optional<torch::Stream>& stream = std::nullopt)
    {
        using namespace celux::conversion; // For IConverter
        // Define whether the device is CPU (true) or CUDA (false)
        bool is_cpu = device.is_cpu();
        bool is_cuda = device.is_cuda();

        // Define the key based on device and pixel format
        ConverterKey key = std::make_tuple(is_cpu, bit_depth, pixfmt);

        // Define the factory map
        static const std::unordered_map<ConverterKey,
                                        std::function<std::unique_ptr<IConverter>(
                                            const std::optional<torch::Stream>&)>,
                                        ConverterKeyHash>
            converterMap = {
                // 8-bit CPU converters
                {std::make_tuple(true, 8, AV_PIX_FMT_YUV420P),
                 [](const std::optional<torch::Stream>&) -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating YUV420PToRGB converter");
                     return std::make_unique<cpu::YUV420PToRGB>();
                 }},
                {std::make_tuple(true, 8, AV_PIX_FMT_RGB24),
				 [](const std::optional<torch::Stream>&) -> std::unique_ptr<IConverter>
				 {
					 CELUX_DEBUG("Creating RGB24ToRGB converter");
					 return std::make_unique<cpu::RGBToYUV420P>();
				 }},
                // 10-bit CPU converters
                {std::make_tuple(true, 10, AV_PIX_FMT_YUV420P10LE),
                 [](const std::optional<torch::Stream>&) -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating YUV420P10ToRGB48 converter");
                     return std::make_unique<cpu::YUV420P10ToRGB48>();
                 }},
#ifdef CUDA_ENABLED
                // 8-bit CUDA converters
                {std::make_tuple(false, 8, AV_PIX_FMT_NV12),
                 [](const std::optional<torch::Stream>& stream)
                     -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating NV12ToRGB converter");
                     return std::make_unique<gpu::cuda::NV12ToRGB>(checkStream(stream));
                 }},
                {std::make_tuple(false, 8, AV_PIX_FMT_RGB24),
				 [](const std::optional<torch::Stream>& stream)
					 -> std::unique_ptr<IConverter>
				 {
					 CELUX_DEBUG("Creating RGB24ToRGB converter");
					 return std::make_unique<gpu::cuda::RGBToYUV420P>(checkStream(stream));
				 }},
                // 10-bit CUDA converters
                {std::make_tuple(false, 10, AV_PIX_FMT_P010LE),
                 [](const std::optional<torch::Stream>& stream) -> std::unique_ptr<IConverter>
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
        throw std::invalid_argument(
            "Unsupported combination - Device: " + deviceType +
            ", Bit Depth: " + std::to_string(bit_depth) +
            ", Pixel Format: " + av_get_pix_fmt_name(pixfmt));
    }
    }; // class Factory

} // namespace celux

#endif // FACTORY_HPP
