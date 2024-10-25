#pragma once

#include "backends/Encoder.hpp"

namespace celux::backends::cpu
{
class Encoder : public celux::Encoder
{
  public:
    Encoder(const std::string& outputPath, int width, int height, double fps,
            celux::EncodingFormats format, const std::string& codecName,
            std::optional<torch::Stream> stream)
        : celux::Encoder(outputPath, width, height, fps, format, codecName, stream)
    {
        this->initialize();
    }

    // No need to override methods unless specific behavior is needed
};
} // namespace celux::backends::cpu
