// Factory.hpp

#pragma once
#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <Decoders.hpp>

namespace celux
{

/**
 * @brief Factory class to create Decoders and Converters based on backend and
 * configuration.
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
    createDecoder(celux::backend backend, const std::string& filename,
                  std::unique_ptr<celux::conversion::IConverter> converter)
    {
        switch (backend)
        {
        case celux::backend::CPU:
            return std::make_unique<celux::backends::cpu::Decoder>(
                filename, std::move(converter));
#ifdef CUDA_ENABLED
        case celux::backend::CUDA:
            return std::make_unique<celux::backends::gpu::cuda::Decoder>(
                filename, std::move(converter));
#endif // CUDA_ENABLED
        default:
            throw std::invalid_argument("Unsupported backend: " +
                                        std::to_string(static_cast<int>(backend)));
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
    static std::unique_ptr<celux::conversion::IConverter>
    createConverter(celux::backend device, celux::ConversionType type,
                    celux::dataType dtype)
    {
        // CPU only supports float32 and uint8
        if (device == celux::backend::CPU && dtype == celux::dataType::FLOAT16)
        {
            throw std::runtime_error("CPU backend does not support half precision");
        }

        // For CPU backend
        if (device == celux::backend::CPU)
        {
            switch (type)
            {
            case celux::ConversionType::NV12ToRGB:
                if (dtype == celux::dataType::UINT8)
                {
                    return std::make_unique<
                        celux::conversion::cpu::NV12ToRGB<uint8_t>>();
                }
                else if (dtype == celux::dataType::FLOAT32)
                {
                    return std::make_unique<celux::conversion::cpu::NV12ToRGB<float>>();
                }
                break;

            default:
                throw std::runtime_error("Unsupported conversion type for CPU backend");
            }
        }

#ifdef CUDA_ENABLED
        // For CUDA backend
        if (device == celux::backend::CUDA)
        {
            switch (type)
            {
            case celux::ConversionType::RGBToNV12:
                if (dtype == celux::dataType::UINT8)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::RGBToNV12<uint8_t>>();
                }
                else if (dtype == celux::dataType::FLOAT16)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::RGBToNV12<half>>();
                }
                else if (dtype == celux::dataType::FLOAT32)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::RGBToNV12<float>>();
                }
                break;

            case celux::ConversionType::NV12ToRGB:
                if (dtype == celux::dataType::UINT8)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::NV12ToRGB<uint8_t>>();
                }
                else if (dtype == celux::dataType::FLOAT16)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::NV12ToRGB<half>>();
                }
                else if (dtype == celux::dataType::FLOAT32)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::NV12ToRGB<float>>();
                }
                break;

            case celux::ConversionType::BGRToNV12:
                if (dtype == celux::dataType::UINT8)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::BGRToNV12<uint8_t>>();
                }
                else if (dtype == celux::dataType::FLOAT16)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::BGRToNV12<half>>();
                }
                else if (dtype == celux::dataType::FLOAT32)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::BGRToNV12<float>>();
                }
                break;

            case celux::ConversionType::NV12ToBGR:
                if (dtype == celux::dataType::UINT8)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::NV12ToBGR<uint8_t>>();
                }
                else if (dtype == celux::dataType::FLOAT16)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::NV12ToBGR<half>>();
                }
                else if (dtype == celux::dataType::FLOAT32)
                {
                    return std::make_unique<
                        celux::conversion::gpu::cuda::NV12ToBGR<float>>();
                }
                break;

            default:
                throw std::runtime_error(
                    "Unsupported conversion type for CUDA backend");
            }
        }
#endif // CUDA_ENABLED

        throw std::runtime_error("Unsupported backend or data type");
    }
};

} // namespace celux

#endif // FACTORY_HPP
