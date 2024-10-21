#pragma once

#include "backends/Encoder.hpp"

namespace celux::backends::cpu
{
class Encoder : public celux::Encoder
{
  public:
    Encoder(const std::string& outputPath, const VideoProperties& props,
            std::unique_ptr<celux::conversion::IConverter> converter = nullptr)
        : celux::Encoder(std::move(converter))
    {
        this->initialize(outputPath, props);
    }

    // No need to override methods unless specific behavior is needed
};
} // namespace celux::backends::cpu
