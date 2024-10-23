#pragma once
#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <Decoders.hpp>
#include <Encoders.hpp> // Assuming you have header files that declare Encoder classes
#include <torch/extension.h>
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
    static std::unique_ptr<Decoder>
    createDecoder(torch::Device device, const std::string& filename,
                  std::unique_ptr<celux::conversion::IConverter> converter)
    {
        if (device.is_cpu())
        {
            return std::make_unique<celux::backends::cpu::Decoder>(
                filename, std::move(converter));
        }
#ifdef CUDA_ENABLED
        else if (device.is_cuda())
        {
            return std::make_unique<celux::backends::gpu::cuda::Decoder>(
                filename, std::move(converter));
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

    /**
     * @brief Creates an IConverter instance based on backend, conversion type, and data
     * type.
     *
     * @param device Backend type (CPU or CUDA).
     * @param type Conversion type (e.g., RGBToNV12).
     * @param dtype Data type (UINT8, FLOAT16, FLOAT32).
     * @return std::unique_ptr<IConverter> Pointer to the created Converter.
     */
static std::unique_ptr<celux::conversion::IConverter> createConverter(
            torch::Device device, celux::ConversionType type,
            std::optional<torch::Stream> stream){
            // For CPU backend
            if (device.is_cpu()){switch (type){
                case celux::ConversionType::NV12ToRGB :

                    return std::make_unique<celux::conversion::cpu::YUV420PToRGB>();

    break;
case celux::ConversionType::RGBToNV12:

    return std::make_unique<celux::conversion::cpu::RGBToYUV420P>();

    break;
default:
    throw std::runtime_error("Unsupported conversion type for CPU backend");
}
}

#ifdef CUDA_ENABLED
// For CUDA backend
if (device.is_cuda())
{
    if (!stream.has_value())
    {
        CELUX_DEBUG("Creating CUDA stream for converter\n");
        stream = c10::cuda::getStreamFromPool(true);
    }
    // make a cuda stream
    c10::cuda::CUDAStream cStream(stream.value());
    switch (type)
    {

    case celux::ConversionType::RGBToNV12:

        return std::make_unique<celux::conversion::gpu::cuda::RGBToYUV420P>(
            cStream.stream());

        break;

    case celux::ConversionType::NV12ToRGB:

        return std::make_unique<celux::conversion::gpu::cuda::NV12ToRGB>(
            cStream.stream());

        break;

    default:
        throw std::runtime_error("Unsupported conversion type for CUDA backend");
    }
}
#endif // CUDA_ENABLED

throw std::runtime_error("Unsupported backend or data type");
}
}
;

} // namespace celux

#endif // FACTORY_HPP
