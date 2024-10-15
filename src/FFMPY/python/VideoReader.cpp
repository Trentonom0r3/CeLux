#include "Python/VideoReader.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

VideoReader::VideoReader(const std::string& filePath, const std::string& device,
                         const std::string& dataType)
    : decoder(nullptr), device(device), currentIndex(0), start_frame(0), end_frame(-1), torchDevice(torch::kCUDA)
{
    try
    {
        if (device == "cuda")
        {
            if (!torch::cuda::is_available())
            {
				throw std::runtime_error("CUDA is not available. Please install a CUDA-enabled version of PyTorch.");
			}
            if (torch::cuda::device_count() == 0)
            {
				throw std::runtime_error("No CUDA devices found. Please check your CUDA installation.");
			}		
		}
        else if (device == "cpu")
        {
            torchDevice = torch::Device(torch::kCPU);
		}
        else
        {
			throw std::invalid_argument("Unsupported device: " + device);
		}
        // Determine the data types based on the dataType argument
        torch::Dtype torchDataType;
        //  py::dtype npDataType;

        if (dataType == "uint8")
        {
            torchDataType = torch::kUInt8;
            //  npDataType = py::dtype::of<uint8_t>();
            convert = std::make_unique<ffmpy::conversion::NV12ToRGB<uint8_t>>();
        }
        else if (dataType == "float32")
        {
            torchDataType = torch::kFloat32;
            // npDataType = py::dtype::of<float>();
            convert = std::make_unique<ffmpy::conversion::NV12ToRGB<float>>();
        }
        else if (dataType == "float16")
        {
            torchDataType = torch::kFloat16;
            // npDataType = py::dtype("float16");
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

        // Initialize numpy buffer with the appropriate data type
        // npBuffer = py::array(npDataType, {properties.height, properties.width, 3});

        // Initialize RGB Tensor on CUDA with the appropriate data type
        RGBTensor = torch::empty(
            {properties.height, properties.width, 3},
            torch::TensorOptions().dtype(torchDataType).device(torch::kCUDA));

        cpuTensor = torch::empty(
            {properties.height, properties.width, 3},
            torch::TensorOptions().dtype(torchDataType).device(torch::kCUDA));
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

void VideoReader::setRange(int start, int end)
{
    start_frame = start;
    end_frame = end;
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
    int result;

    // Release GIL during decoding
    {
        py::gil_scoped_release release;
        result = decoder->decodeNextFrame(RGBTensor.data_ptr());
    }

    if (result == 1) // Frame decoded successfully
    {
        cpuTensor.copy_(RGBTensor, /*non_blocking=*/true);
        if (torchDevice == torch::kCPU)
        {
            {
                py::gil_scoped_release release;
            }
            return py::cast(cpuTensor.to(torch::kCPU, /*non_blocking=*/true));
        }
        else
        {
            return py::cast(cpuTensor);
        }
    }
    else if (result == 0) // End of video stream
    {
        throw py::stop_iteration();
    }
    else // Decoding failed
    {
        throw std::runtime_error("Failed to decode the next frame.");
    }
}

bool VideoReader::seek(double timestamp)
{
    bool success;

    // Release GIL during seeking
    {
        py::gil_scoped_release release;
        success = decoder->seek(timestamp);
    }

    return success;
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
    props["total_frames"] = properties.totalFrames;
    props["pixel_format"] = av_get_pix_fmt_name(properties.pixelFormat);
    props["has_audio"] = properties.hasAudio;
    return props;
}

void VideoReader::reset()
{
    seek(0.0); // Reset to the beginning
    currentIndex = 0;
}

bool VideoReader::seekToFrame(int frame_number)
{
    if (frame_number < 0 || frame_number >= properties.totalFrames)
    {
        return false; // Out of range
    }

    // Convert frame number to timestamp in seconds
    double timestamp = static_cast<double>(frame_number) / properties.fps;

    return seek(timestamp);
}

VideoReader& VideoReader::iter()
{
    currentIndex = start_frame;
    seekToFrame(start_frame);
    return *this;
}

py::object VideoReader::next()
{
    if (end_frame >= 0 && currentIndex > end_frame)
    {
        throw py::stop_iteration(); // Stop iteration if range is exhausted
    }

    py::object frame = readFrame();
    if (frame.is_none())
    {
        throw py::stop_iteration(); // Stop iteration if no more frames are available
    }

    currentIndex++;
    return frame;
}

void VideoReader::enter()
{
    // Resources are already initialized in the constructor
}

void VideoReader::exit(const py::object& exc_type, const py::object& exc_value,
                       const py::object& traceback)
{
    close(); // Close the video reader and free resources
}

void VideoReader::copyTo(void* src, void* dst, size_t size, CopyType type)
{
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        // Reacquire GIL before throwing exception
        py::gil_scoped_acquire acquire;
        throw std::runtime_error("Error copying data to host: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

int VideoReader::length() const
{
    return properties.totalFrames;
}

void VideoReader::sync()
{
    // Release GIL during synchronization
    {
        py::gil_scoped_release release;
        if (convert)
        {
            convert->synchronize();
        }
    }
}
