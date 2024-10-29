// CPU Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace celux::backends::cpu
{
class Decoder : public celux::Decoder
{
  public:
    Decoder(const std::string& filePath, std::optional<torch::Stream> stream,
            int numThreads)
        : celux::Decoder(stream, numThreads)
    {
        isHwAccel = false;
        initialize(filePath);
    }

    // No need to override methods unless specific behavior is needed
};
} // namespace celux::backends::cpu
