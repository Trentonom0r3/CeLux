// CUDA Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace celux::backends::gpu::cuda
{
class Decoder : public celux::Decoder
{
  public:
    Decoder(const std::string& filePath, int numThreads,
            std::vector<std::shared_ptr<FilterBase>> filters)
        : celux::Decoder(numThreads, filters)
    {
        isHwAccel = true;
        initialize(filePath);
    }

  protected:
    void initHWAccel() override;
};

} // namespace celux::backends::gpu::cuda
