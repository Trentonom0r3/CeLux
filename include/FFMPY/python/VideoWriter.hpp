// VideoWriter
#ifndef VIDEOWRITER_HPP
#define VIDEOWRITER_HPP

#include "Encoder.hpp"
#include "NV12ToRGB.hpp"
#include "RGBToNV12.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Enum for copy types
enum class CopyType
{
    HOST,
    DEVICE
};

//py::dict config = py::dict();
//config["width"] = 1920;
//config["height"] = 1080;
//config["fps"] = 30;
//config["as_numpy"] = true;
//config["dtype"] = "uint8";

py::dict createConfig(int width, int height, int fps, bool as_numpy, std::string dtype)
{
	py::dict config;
	config["width"] = width;
	config["height"] = height;
	config["fps"] = fps;
	config["as_numpy"] = as_numpy;
	config["dtype"] = dtype;
	return config;
}

class VideoWriter
{
  public:
    VideoWriter(const std::string& filePath, py::dict config);

    ~VideoWriter();

    bool writeFrame(py::object frame); //will ahve to determine what type of object frame is, tensor vs numpy array, uint8_t vs float32, float16

    std::vector<std::string> supportedCodecs();

  private:

    void copyTo(void* src, void* dst, size_t size, CopyType type);

    /**
     * @brief Close the video writer and release resources.
     */
    void close();

    // Member variables
    std::unique_ptr<ffmpy::Encoder> encoder;

    std::unique_ptr<ffmpy::conversion::IConverter> convert;

    bool as_numpy;
    void* npBuffer;

};

#endif // VideoWriter_HPP
