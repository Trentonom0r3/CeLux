// CPU Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace ffmpy::backends::cpu
{
class Decoder : public ffmpy::Decoder
{
  public:
    Decoder(const std::string& filePath,
            std::unique_ptr<ffmpy::conversion::IConverter> converter = nullptr)
        : ffmpy::Decoder(std::move(converter))
    {
        initialize(filePath);
    }

    ~Decoder() override
    {
    	// Cleanup
	}

    // No need to override methods unless specific behavior is needed
};
} // namespace ffmpy::backends::cpu
