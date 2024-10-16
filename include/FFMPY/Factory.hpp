#pragma once
#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <Decoders.hpp>

// ffmpy::backend is enum class
static std::unique_ptr<ffmpy::Decoder>
createDecoder(ffmpy::backend backend, const std::string& filename,
              std::unique_ptr<ffmpy::conversion::IConverter> converter)
{
    switch (backend)
    {
    case ffmpy::backend::CPU:
        return std::make_unique<ffmpy::backends::cpu::Decoder>(filename,
                                                               std::move(converter));
#ifdef CUDA_ENABLED
    case ffmpy::backend::CUDA:
        return std::make_unique<ffmpy::backends::gpu::cuda::Decoder>(
            filename, std::move(converter));
#endif // CUDA_ENABLED
    }
}
/*namespace ffmpy
{
enum class backend
{
    CPU,
    CUDA
};
enum class ConversionType
{
    RGBToNV12,
    NV12ToRGB,
    BGRToNV12,
    NV12ToBGR,
};

enum class dataType
{
    UINT8,
    FLOAT16,
    FLOAT32,
};
*/

static std::unique_ptr<ffmpy::conversion::IConverter>
createConverter(ffmpy::backend device, ffmpy::ConversionType type,
                ffmpy::dataType dtype)
{
    // cpu only has float32 and uint8, throw err if device and dtype are cpu and half
    if (device == ffmpy::backend::CPU && dtype == ffmpy::dataType::FLOAT16)
    {
        throw std::runtime_error("CPU backend does not support half precision");
    }
    // For CPU backend
    if (device == ffmpy::backend::CPU)
    {
        switch (type)
        {

        case ffmpy::ConversionType::NV12ToRGB:
            if (dtype == ffmpy::dataType::UINT8)
            {
                return std::make_unique<ffmpy::conversion::cpu::NV12ToRGB<uint8_t>>();
            }
            else if (dtype == ffmpy::dataType::FLOAT32)
            {
                return std::make_unique<ffmpy::conversion::cpu::NV12ToRGB<float>>();
            }
            break;

        default:
            throw std::runtime_error("Unsupported conversion type for CPU backend");
        }
    }
#ifdef CUDA_ENABLED
    // For CUDA backend
    if (device == ffmpy::backend::CUDA)
    {
        switch (type)
        {
        case ffmpy::ConversionType::RGBToNV12:
            if (dtype == ffmpy::dataType::UINT8)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::RGBToNV12<uint8_t>>();
            }
            else if (dtype == ffmpy::dataType::FLOAT16)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::RGBToNV12<half>>();
            }
            else if (dtype == ffmpy::dataType::FLOAT32)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::RGBToNV12<float>>();
            }
            break;

        case ffmpy::ConversionType::NV12ToRGB:
            if (dtype == ffmpy::dataType::UINT8)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::NV12ToRGB<uint8_t>>();
            }
            else if (dtype == ffmpy::dataType::FLOAT16)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::NV12ToRGB<half>>();
            }
            else if (dtype == ffmpy::dataType::FLOAT32)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::NV12ToRGB<float>>();
            }
            break;

        case ffmpy::ConversionType::BGRToNV12:
            if (dtype == ffmpy::dataType::UINT8)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::BGRToNV12<uint8_t>>();
            }
            else if (dtype == ffmpy::dataType::FLOAT16)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::BGRToNV12<half>>();
            }
            else if (dtype == ffmpy::dataType::FLOAT32)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::BGRToNV12<float>>();
            }
            break;

        case ffmpy::ConversionType::NV12ToBGR:
            if (dtype == ffmpy::dataType::UINT8)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::NV12ToBGR<uint8_t>>();
            }
            else if (dtype == ffmpy::dataType::FLOAT16)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::NV12ToBGR<half>>();
            }
            else if (dtype == ffmpy::dataType::FLOAT32)
            {
                return std::make_unique<
                    ffmpy::conversion::gpu::cuda::NV12ToBGR<float>>();
            }
            break;

        default:
            throw std::runtime_error("Unsupported conversion type for CUDA backend");
        }
    }
#endif // CUDA_ENABLED
}

#endif
