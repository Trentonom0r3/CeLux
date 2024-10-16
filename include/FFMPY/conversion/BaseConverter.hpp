// ConverterBase.hpp

#pragma once

#include "IConverter.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace ffmpy
{
namespace conversion
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
        cudaError_t err = cudaStreamCreate(&conversionStream);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }
}

// Destructor
template <typename T> ConverterBase<T>::~ConverterBase()
{
    if (conversionStream)
    {
        synchronize();
        cudaStreamDestroy(conversionStream);
    }
}

// Synchronize Method
template <typename T> void ConverterBase<T>::synchronize()
{
    cudaError_t err = cudaStreamSynchronize(conversionStream);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to synchronize CUDA stream");
    }
}

// Get Stream Method
template <typename T> cudaStream_t ConverterBase<T>::getStream() const
{
    return conversionStream;
}

} // namespace conversion
} // namespace ffmpy
