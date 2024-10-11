// VideoReader.cpp

#include "Python/VideoReader.hpp"
#include <torch/extension.h>

VideoReader::VideoReader(const std::string& filePath, bool as_numpy,
                                 const std::string& dataType) //converter arg to pass in --- enum of optios? NV12To...?
    : decoder(nullptr), as_numpy(as_numpy),
      currentIndex(0)
{
    try
    {
        // Determine the data types based on the dataType argument
        torch::Dtype torchDataType;
        py::dtype npDataType;

        if (dataType == "uint8")
        {
            torchDataType = torch::kUInt8;
            npDataType = py::dtype::of<uint8_t>();
            convert = std::make_unique<ffmpy::conversion::NV12ToRGB<uint8_t>>();
        }
        else if (dataType == "float32")
        {
            torchDataType = torch::kFloat32;
            npDataType = py::dtype::of<float>();
            convert = std::make_unique<ffmpy::conversion::NV12ToRGB<float>>();
        }
        else if (dataType == "float16")
        {
            torchDataType = torch::kFloat16;
            npDataType = py::dtype("float16");
            convert = std::make_unique<ffmpy::conversion::NV12ToRGB<__half>>();
        }
        else
        {
            throw std::invalid_argument("Unsupported dataType: " + dataType);
        }
        bool useHardware = true;
        std::string hwType = "cuda";
        decoder = std::make_unique<ffmpy::Decoder>(filePath, useHardware, hwType,
													   std::move(convert));
        properties = decoder->getVideoProperties();
        if (as_numpy)
        {
          // Initialize numpy buffer with the appropriate data type
          npBuffer = py::array(npDataType, {properties.height, properties.width, 3});
        }
        else
        {
          // Initialize RGB Tensor on CUDA with the appropriate data type
          RGBTensor = torch::empty(
              {properties.height, properties.width, 3},
              torch::TensorOptions().dtype(torchDataType).device(torch::kCUDA));
        }
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
    if (convert)
    {
        convert->synchronize();
        convert.reset();
    }
    if (decoder)
    {
        decoder->close(); // Assuming Decoder has a close method
        decoder.reset();
    }
}

py::object VideoReader::readFrame()
{
    if (decoder->decodeNextFrame(RGBTensor.data_ptr()))
    { // Frame decoded successfully

        if (!as_numpy)
        {
            return py::cast(RGBTensor); // User wants Torch Tensor output
        }

        else
        {
            size_t size = RGBTensor.nbytes(); // Get the size in bytes

            copyTo(RGBTensor.data_ptr(), npBuffer.mutable_data(), size,
                   CopyType::HOST);

            return npBuffer;
        }
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
    // props["codec"] = properties.codec; - not necessary ig
    props["audio"] = properties.audio; // returns a bool value
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

void VideoReader::copyTo(void* src, void* dst, size_t size, CopyType type)
{
    cudaError_t err;
    err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, convert->getStream());
}

int VideoReader::length() const
{
    return properties.totalFrames;
}
