// ConverterBase.hpp

#pragma once

#include "IConverter.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace celux
{
namespace conversion
{
namespace gpu
{
namespace cuda
{

template <typename T> class ConverterBase : public IConverter
{
  public:
    ConverterBase();
    ConverterBase(cudaStream_t stream);
    virtual ~ConverterBase();

    virtual void synchronize() override;
    virtual cudaStream_t getStream() const;

  protected:
    cudaStream_t conversionStream;
    bool passedInStream = false;
};

// Template Definitions

// Default Constructor
template <typename T> ConverterBase<T>::ConverterBase()
{
    cudaError_t err = cudaStreamCreate(&conversionStream);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to create CUDA stream");
    }
}

// Constructor with Stream Parameter
template <typename T>
ConverterBase<T>::ConverterBase(cudaStream_t stream) : conversionStream(stream)
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
}

// Destructor
template <typename T> ConverterBase<T>::~ConverterBase()
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

// Synchronize Method
template <typename T> void ConverterBase<T>::synchronize()
{
    CELUX_DEBUG("Synchronizing CUDA Stream");
    cudaError_t err = cudaStreamSynchronize(conversionStream);
    if (err != cudaSuccess)
    {
        CELUX_DEBUG("Failed to synchronize CUDA stream");
        throw std::runtime_error("Failed to synchronize CUDA stream");
    }
}

// Get Stream Method
template <typename T> cudaStream_t ConverterBase<T>::getStream() const
{
    return conversionStream;
}

} // namespace cuda
} // namespace gpu
} // namespace conversion
} // namespace celux
