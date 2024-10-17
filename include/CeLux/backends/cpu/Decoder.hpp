// CPU Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace celux::backends::cpu
{
class Decoder : public celux::Decoder
{
  public:
    Decoder(const std::string& filePath,
            std::unique_ptr<celux::conversion::IConverter> converter = nullptr)
        : celux::Decoder(std::move(converter))
    {
        initialize(filePath);
    }

    ~Decoder() override
    {
    	// Cleanup
	}

    // No need to override methods unless specific behavior is needed
};
} // namespace celux::backends::cpu
