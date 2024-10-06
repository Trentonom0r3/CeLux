#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

namespace ffmpy
{
namespace conversion
{
template <typename T> class ConverterBase
{
  public:
    ConverterBase();
    ConverterBase(cudaStream_t stream);
    ~ConverterBase();

    void synchronize();
    cudaStream_t getStream();

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
template <typename T> cudaStream_t ConverterBase<T>::getStream()
{
    return conversionStream;
}
} // namespace conversions
} // namespace ffmpy
