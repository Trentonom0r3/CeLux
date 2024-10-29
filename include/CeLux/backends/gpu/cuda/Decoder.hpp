// CUDA Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace celux::backends::gpu::cuda
{
class Decoder : public celux::Decoder
{
  public:
    Decoder(const std::string& filePath, std::optional<torch::Stream> stream, int numThreads)
        : celux::Decoder(stream, numThreads)
    {
        isHwAccel = true;
        initialize(filePath);
    }

  protected:
    void initHWAccel() override;
};

} // namespace celux::backends::gpu::cuda
