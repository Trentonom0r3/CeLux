// VideoWriter
#ifndef VIDEOWRITER_HPP
#define VIDEOWRITER_HPP

#include "Encoder.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class VideoWriter
{
  public:
    VideoWriter(const std::string& filePath, int width, int height, float fps,
                bool as_numpy, std::string dtype);

    ~VideoWriter();

    bool writeFrame(
        py::object frame); // will ahve to determine what type of object frame is,
                           // tensor vs numpy array, uint8_t vs float32, float16

    std::vector<std::string> supportedCodecs();

    /**
     * @brief Close the video writer and release resources.
     */
    void close();

  private:
    void copyTo(void* src, void* dst, size_t size);

    // Member variables
    std::unique_ptr<ffmpy::Encoder> encoder;

    std::unique_ptr<ffmpy::conversion::IConverter> convert;

    bool as_numpy;
    void* npBuffer;
};

#endif // VideoWriter_HPP
