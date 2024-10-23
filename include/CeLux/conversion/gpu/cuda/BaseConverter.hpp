// ConverterBase.hpp

#pragma once

#include "IConverter.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <nppi_color_conversion.h>

namespace celux
{
namespace conversion
{
namespace gpu
{
namespace cuda
{

class ConverterBase : public IConverter
{
  public:
    ConverterBase();
    ConverterBase(cudaStream_t stream);
    virtual ~ConverterBase();

    virtual void synchronize() override;
    virtual cudaStream_t getStream() const;

  protected:
    cudaStream_t conversionStream;
    NppStreamContext nppStreamContext;
    bool passedInStream = false;
};


inline ConverterBase::ConverterBase()
{
    cudaError_t err = cudaStreamCreate(&conversionStream);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to create CUDA stream");
    }
    nppStreamContext.hStream = this->conversionStream;
}


inline ConverterBase::ConverterBase(cudaStream_t stream) : conversionStream(stream)
{
    if (stream == nullptr)
    {
        CELUX_DEBUG("Stream was nullptr, creating new Stream");
        cudaError_t err = cudaStreamCreate(&conversionStream);
        if (err != cudaSuccess)
        {
            CELUX_CRITICAL("Failed to create CUDA stream in ConverterBase");
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }
    else
    {
		passedInStream = true;
	}
    nppStreamContext.hStream = this->conversionStream;
}

inline ConverterBase::~ConverterBase()
{
    CELUX_DEBUG("Destroying ConverterBase");
    if (conversionStream)
    {
        synchronize();
        if (!passedInStream)
        {
            CELUX_DEBUG("Destroying CUDA Stream");
            cudaStreamDestroy(conversionStream);
        }
    }
}

inline void ConverterBase::synchronize()
{
    CELUX_DEBUG("Synchronizing CUDA Stream");
    cudaError_t err = cudaStreamSynchronize(conversionStream);
    if (err != cudaSuccess)
    {
        CELUX_DEBUG("Failed to synchronize CUDA stream");
        throw std::runtime_error("Failed to synchronize CUDA stream");
    }
}

inline cudaStream_t ConverterBase::getStream() const
{
    return conversionStream;
}

} // namespace cuda
} // namespace gpu
} // namespace conversion
} // namespace celux
