// ConverterBase.hpp

#pragma once

#include "IConverter.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nppi_color_conversion.h>
#include <type_traits>

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
    virtual ~ConverterBase();

    virtual void synchronize() override;
    virtual cudaStream_t getStream() const;

  protected:
    cudaStream_t conversionStream;
    cudaEvent_t conversionEvent; // CUDA event for synchronization
    NppStreamContext nppStreamContext;
};

inline ConverterBase::ConverterBase()
{
    cudaError_t err = cudaStreamCreate(&conversionStream);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to create CUDA stream");
    }

    // Create a CUDA event with default flags
    err = cudaEventCreate(&conversionEvent);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to create CUDA event");
    }

    nppStreamContext.hStream = this->conversionStream;
}

inline ConverterBase::~ConverterBase()
{
    CELUX_DEBUG("Destroying ConverterBase");

    // Ensure synchronization before cleanup
    synchronize();

    // Destroy the event
    cudaEventDestroy(conversionEvent);

    if (conversionStream)
    {

        CELUX_DEBUG("Destroying CUDA Stream");
        cudaStreamDestroy(conversionStream);
    }
}

inline void ConverterBase::synchronize()
{
    CELUX_DEBUG("Synchronizing CUDA Stream with Event");

    // Record the event on the current stream
    cudaError_t err = cudaEventRecord(conversionEvent, conversionStream);
    if (err != cudaSuccess)
    {
        CELUX_DEBUG("Failed to record CUDA event");
        throw std::runtime_error("Failed to record CUDA event");
    }

    // Wait for the event to complete
    err = cudaEventSynchronize(conversionEvent);
    if (err != cudaSuccess)
    {
        CELUX_DEBUG("Failed to synchronize on CUDA event");
        throw std::runtime_error("Failed to synchronize on CUDA event");
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
