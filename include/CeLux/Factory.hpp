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
    static std::unique_ptr<Decoder> createDecoder(torch::Device device, const std::string& filename, int numThreads,
                  std::vector<std::shared_ptr<FilterBase>> filters)
    {

            return std::make_unique<celux::backends::cpu::Decoder>(filename, numThreads,
                                                                   filters);
      
    }


    /**
     * @brief Creates a Converter instance based on the specified backend and pixel
     * format.
     *
     * @param device Device type (CPU or CUDA).
     * @param pixfmt Pixel format.
     * @param  Optional  for CUDA operations.
     * @return std::unique_ptr<celux::conversion::IConverter> Pointer to the created
     * Converter.
     */
    static std::unique_ptr<celux::conversion::IConverter>
    createConverter(const torch::Device& device, AVPixelFormat pixfmt)
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
                                            )>,
                                        ConverterKeyHash>
            converterMap = {
                // CPU converters
                {std::make_tuple(true, AV_PIX_FMT_YUV420P),
                 []() -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating YUV420PToRGB converter");
                     return std::make_unique<cpu::YUV420PToRGB>();
                 }},

                {std::make_tuple(true, AV_PIX_FMT_YUV420P10LE),
                 []() -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating YUV420P10LEToRGB48 converter");
                     return std::make_unique<cpu::YUV420P10ToRGB48>();
                 }},
                {std::make_tuple(true, AV_PIX_FMT_YUV422P10LE),
				 []() -> std::unique_ptr<IConverter>
				 {
					 CELUX_DEBUG("Creating YUV422P10LEToRGB48 converter");
					 return std::make_unique<cpu::YUV422P10ToRGB48>();
				 }},

                // bgr24
                {std::make_tuple(true, AV_PIX_FMT_BGR24),
                 []() -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating BGR24ToRGB converter");
                     return std::make_unique<cpu::BGRToRGB>();
                 }},
                //rgb24
                {std::make_tuple(true, AV_PIX_FMT_RGB24),
                         []() -> std::unique_ptr<IConverter>
                         {
					 CELUX_DEBUG("Creating RGB24ToRGB converter");
					 return std::make_unique<cpu::RGBToRGB>();
				 }},
                // New CPU converters with alpha channels
                {std::make_tuple(true, AV_PIX_FMT_RGBA),
                 []() -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating RGBAToRGB converter");
                     return std::make_unique<cpu::RGBAToRGB>();
                 }},
                {std::make_tuple(true, AV_PIX_FMT_BGRA),
                 []() -> std::unique_ptr<IConverter>
                 {
                     CELUX_DEBUG("Creating BGRAToRGB converter");
                     return std::make_unique<cpu::BGRAToRGB>();
                 }},
                {std::make_tuple(true, AV_PIX_FMT_GBRP),
				 []() -> std::unique_ptr<IConverter>
				 {
					 CELUX_DEBUG("Creating GBRPToRGB converter");
					 return std::make_unique<cpu::GBRPToRGB>();
				 }},
                
            };

        // Search for the converter in the map
        auto it = converterMap.find(key);
        if (it != converterMap.end())
        {
            return it->second();
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
        case AV_PIX_FMT_RGBA:
        case AV_PIX_FMT_BGRA:
        case AV_PIX_FMT_GBRP:
            return 8;
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_YUV422P10LE:
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
