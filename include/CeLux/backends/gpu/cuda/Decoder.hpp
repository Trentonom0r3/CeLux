// CUDA Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace celux::backends::gpu::cuda
{
class Decoder : public celux::Decoder
{
  public:
    Decoder(const std::string& filePath, std::optional<torch::Stream> stream)
        : celux::Decoder(celux::Decoder())
    {
        isHwAccel = true;
        initialize(filePath);
    }

  protected:
    void initHWAccel() override;
};

} // namespace celux::backends::gpu::cuda
