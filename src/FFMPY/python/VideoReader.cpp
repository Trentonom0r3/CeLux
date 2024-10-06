// VideoReader.cpp

#include "Python/VideoReader.hpp"
#include <torch/extension.h>

VideoReader::VideoReader(const std::string& filePath, bool useHardware,
                         const std::string& hwType, bool as_numpy)
    : decoder(std::make_unique<ffmpy::Decoder>(filePath, useHardware, hwType)),
      properties(decoder->getVideoProperties()), as_numpy(as_numpy),
      convert(ffmpy::conversion::NV12ToRGB<uint8_t>()), currentIndex(0)
{
    try
    {
        // Initialize RGB Tensor on CUDA by default
        rgb_tensor = torch::empty(
            {properties.height, properties.width, 3},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

        npBuffer = py::array_t<uint8_t>({properties.height, properties.width, 3});
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Exception in VideoReader constructor: " << ex.what() << std::endl;
        throw; // Re-throw exception after logging
    }
}

VideoReader::~VideoReader()
{
    close();
}

void VideoReader::close()
{
    if (decoder)
    {
        decoder->close(); // Assuming Decoder has a close method
        decoder.reset();
    }
    // Additional resource cleanup if necessary
}

py::object VideoReader::readFrame()
{
    if (decoder->decodeNextFrame(frame))
    { // Frame decoded successfully
        // Convert frame to RGB
        convert.convert(frame, rgb_tensor.data_ptr<uint8_t>());

        if (as_numpy)
        { // User wants NumPy array

            // Data is already on CPU
            copyTo(rgb_tensor.data_ptr<uint8_t>(), npBuffer.mutable_data(),
                   npBuffer.size(), CopyType::HOST);

            return npBuffer;
        }
        else
            return py::cast(rgb_tensor);
    }
    else
    { // No frame decoded
        return py::none();
    }
}

bool VideoReader::seek(double timestamp)
{
    return decoder->seek(timestamp);
}

std::vector<std::string> VideoReader::supportedCodecs()
{
    return decoder->listSupportedDecoders();
}

py::dict VideoReader::getProperties() const
{
    py::dict props;
    props["width"] = properties.width;
    props["height"] = properties.height;
    props["fps"] = properties.fps;
    props["duration"] = properties.duration;
    props["totalFrames"] = properties.totalFrames;
    props["pixelFormat"] = av_get_pix_fmt_name(properties.pixelFormat);
    return props;
}

void VideoReader::reset()
{
    seek(0.0); // Reset to the beginning
    currentIndex = 0;
}

VideoReader& VideoReader::iter()
{
    // reset(); // Reset to the beginning
    return *this;
}

py::object VideoReader::next()
{
    if (currentIndex >= properties.totalFrames)
    {
        throw py::stop_iteration(); // Signal the end of iteration
    }
    auto frame_obj = readFrame();
    if (frame_obj.is_none())
    {
        throw py::stop_iteration(); // Stop iteration if frame couldn't be read
    }
    currentIndex++;
    return frame_obj;
}

void VideoReader::enter()
{
    // Could be used to initialize or allocate resources
    // Currently, all resources are initialized in the constructor
}

void VideoReader::exit(const py::object& exc_type, const py::object& exc_value,
                       const py::object& traceback)
{
    close(); // Close the video reader and free resources
}

void VideoReader::copyTo(uint8_t* src, uint8_t* dst, size_t size, CopyType type)
{
    cudaError_t err;
    err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, convert.getStream());
}

int VideoReader::length() const
{
    return properties.totalFrames;
}
