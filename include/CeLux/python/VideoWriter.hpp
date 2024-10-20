// VideoWriter
#ifndef VIDEOWRITER_HPP
#define VIDEOWRITER_HPP

#include "Factory.hpp"
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class VideoWriter
{
  public:
    VideoWriter(const std::string& filePath, int width, int height, float fps,
        const std::string& device,
        const std::string& d_type, std::optional<torch::Stream> stream = std::nullopt);

    ~VideoWriter();

    bool writeFrame(torch::Tensor frame); // will ahve to determine what type of object frame is,
                           // tensor vs numpy array, uint8_t vs float32, float16

    std::vector<std::string> supportedCodecs();

    /**
     * @brief Close the video writer and release resources.
     */
    void close();

  private:
    std::unique_ptr<celux::Encoder> encoder;

    std::unique_ptr<celux::conversion::IConverter> convert;
    torch::Tensor intermediateFrame;
};

#endif // VideoWriter_HPP
