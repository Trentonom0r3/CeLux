// Python/VideoReader.cpp

#include "Python/VideoReader.hpp"
#include <pybind11/pybind11.h>
#include <torch/torch.h> // Ensure you have included the necessary Torch headers

namespace py = pybind11;

VideoReader::VideoReader(const std::string& filePath, const std::string& device,
                         std::optional<torch::Stream> stream)
    : decoder(nullptr), currentIndex(0), start_frame(0), end_frame(-1)
{
    CELUX_INFO("VideoReader constructor called with filePath: {}", filePath);
    CELUX_INFO("Device: {}", device);

    try
    {
        torch::Device torchDevice = torch::Device(torch::kCPU);
        CELUX_INFO("Creating VideoReader instance");
        if (device == "cuda")
        {
            CELUX_DEBUG("Device is set to CUDA");
            if (!torch::cuda::is_available())
            {
                CELUX_CRITICAL("CUDA is not available. Please install a CUDA-enabled "
                               "version of celux.");
                throw std::runtime_error("CUDA is not available. Please install a "
                                         "CUDA-enabled version of celux.");
            }
            if (torch::cuda::device_count() == 0)
            {
                CELUX_CRITICAL(
                    "No CUDA devices found. Please check your CUDA installation.");
                throw std::runtime_error(
                    "No CUDA devices found. Please check your CUDA installation.");
            }
            torchDevice = torch::Device(torch::kCUDA);
        }
        else if (device == "cpu")
        {
            CELUX_DEBUG("Device is set to CPU");
            CELUX_TRACE("Using CPU device for VideoReader");
            torchDevice = torch::Device(torch::kCPU);
        }
        else
        {
            CELUX_CRITICAL("Unsupported device: {}", device);
            throw std::invalid_argument("Unsupported device: " + device);
        }

        torch::Dtype torchDataType;

        torchDataType = torch::kUInt8;
        CELUX_DEBUG("DataType mapped to UINT8");

        decoder =
            celux::Factory::createDecoder(torchDevice, filePath, stream);
        CELUX_DEBUG("Decoder created successfully");

        // Retrieve video properties
        properties = decoder->getVideoProperties();
        CELUX_INFO("Video properties retrieved: width={}, height={}, fps={}, "
                   "duration={}, totalFrames={}, pixelFormat={}, hasAudio={}",
                   properties.width, properties.height, properties.fps,
                   properties.duration, properties.totalFrames,
                   av_get_pix_fmt_name(properties.pixelFormat), properties.hasAudio);

        // Initialize tensor
        tensor = torch::empty(
            {properties.height, properties.width, 3},
            torch::TensorOptions().dtype(torchDataType).device(torchDevice));

        CELUX_DEBUG("Torch tensor initialized with shape: [{}, {}, {}] :, "
                    "device: {}",
                    properties.height, properties.width, 3, device);
    }
    catch (const std::exception& ex)
    {
        CELUX_ERROR("Exception in VideoReader constructor: {}", ex.what());
        throw; // Re-throw exception after logging
    }
}

VideoReader::~VideoReader()
{
    CELUX_INFO("VideoReader destructor called");
    close();
}

void VideoReader::setRange(int start, int end)
{
    CELUX_INFO("Setting frame range: start={}, end={}", start, end);
    // Handle negative indices by converting them to positive frame numbers
    if (start < 0)
    {
        start = properties.totalFrames + start;
        CELUX_DEBUG("Adjusted start_frame to {}", start);
    }
    if (end < 0)
    {
        end = properties.totalFrames + end;
        CELUX_DEBUG("Adjusted end_frame to {}", end);
    }

    // Validate the adjusted frame range
    if (start < 0 || end < 0)
    {
        CELUX_ERROR("Frame indices out of range after adjustment: start={}, end={}",
                    start, end);
        throw std::runtime_error("Frame indices out of range after adjustment.");
    }
    if (end <= start)
    {
        CELUX_ERROR("Invalid frame range: end_frame ({}) must be greater than "
                    "start_frame ({}) after adjustment.",
                    end, start);
        throw std::runtime_error(
            "end_frame must be greater than start_frame after adjustment.");
    }

    // Make end_frame exclusive by subtracting one
    end = end - 1;
    CELUX_DEBUG("Adjusted end_frame to be exclusive: {}", end);

    start_frame = start;
    end_frame = end;
    CELUX_INFO("Frame range set: start_frame={}, end_frame={}", start_frame, end_frame);
}

torch::Tensor VideoReader::readFrame()
{
    CELUX_TRACE("readFrame() called");
    py::gil_scoped_release release; // Release GIL before calling decoder
    bool success = decoder->decodeNextFrame(tensor.data_ptr());
    if (!success)
    {
        CELUX_WARN("Decoding failed or no more frames available");
        return torch::Tensor(); // Return an empty tensor if decoding failed
    }

    CELUX_DEBUG("Frame decoded successfully");
    return tensor;
}

void VideoReader::close()
{
    CELUX_INFO("Closing VideoReader");
    // Clean up decoder and other resources
    if (decoder)
    {
        CELUX_DEBUG("Closing decoder");
        decoder->close();
        decoder.reset();
        CELUX_INFO("Decoder closed and reset successfully");
    }
    else
    {
        CELUX_DEBUG("Decoder already closed or was never initialized");
    }
}

bool VideoReader::seek(double timestamp)
{
    CELUX_INFO("Seeking to timestamp: {}", timestamp);
    bool success;

    // Release GIL during seeking
    {
        // py::gil_scoped_release release; // Uncomment if GIL management is needed
        success = decoder->seek(timestamp);
    }

    if (success)
    {
        CELUX_DEBUG("Seek to timestamp {} successful", timestamp);
    }
    else
    {
        CELUX_WARN("Seek to timestamp {} failed", timestamp);
    }

    return success;
}

std::vector<std::string> VideoReader::supportedCodecs()
{
    CELUX_TRACE("supportedCodecs() called");
    std::vector<std::string> codecs = decoder->listSupportedDecoders();
    CELUX_DEBUG("Number of supported decoders: {}", codecs.size());
    for (const auto& codec : codecs)
    {
        CELUX_TRACE("Supported decoder: {}", codec);
    }
    return codecs;
}

py::dict VideoReader::getProperties() const
{
    CELUX_TRACE("getProperties() called");
    py::dict props;
    props["width"] = properties.width;
    props["height"] = properties.height;
    props["fps"] = properties.fps;
    props["duration"] = properties.duration;
    props["total_frames"] = properties.totalFrames;
    props["pixel_format"] = av_get_pix_fmt_name(properties.pixelFormat)
                                ? av_get_pix_fmt_name(properties.pixelFormat)
                                : "Unknown";
    props["has_audio"] = properties.hasAudio;
    CELUX_DEBUG("Video properties retrieved and converted to Python dict");
    return props;
}

void VideoReader::reset()
{
    CELUX_INFO("Resetting VideoReader to the beginning");
    bool success = seek(0.0);
    if (success)
    {
        currentIndex = 0;
        CELUX_DEBUG("VideoReader reset successfully");
    }
    else
    {
        CELUX_WARN("Failed to reset VideoReader to the beginning");
    }
}

bool VideoReader::seekToFrame(int frame_number)
{
    CELUX_INFO("Seeking to frame number: {}", frame_number);
    if (frame_number < 0 || frame_number > properties.totalFrames)
    {
        CELUX_ERROR("Frame number {} is out of range (0 to {})", frame_number,
                    properties.totalFrames);
        return false; // Out of range
    }

    // Convert frame number to timestamp in seconds
    double timestamp = static_cast<double>(frame_number) / properties.fps;
    CELUX_DEBUG("Converted frame number {} to timestamp {} seconds", frame_number,
                timestamp);

    bool success = seek(timestamp);
    if (success)
    {
        CELUX_DEBUG("Seek to frame number {} successful", frame_number);
    }
    else
    {
        CELUX_WARN("Seek to frame number {} failed", frame_number);
    }

    return success;
}

VideoReader& VideoReader::iter()
{
    CELUX_TRACE("iter() called: Preparing VideoReader for iteration");
    currentIndex = start_frame;
    bool success = seekToFrame(start_frame);
    if (success)
    {
        CELUX_DEBUG("VideoReader successfully seeked to start_frame: {}", start_frame);
    }
    else
    {
        CELUX_WARN("Failed to seek to start_frame: {}", start_frame);
    }
    return *this;
}

torch::Tensor VideoReader::next()
{
    CELUX_TRACE("next() called: Retrieving next frame");
    if (end_frame >= 0 && currentIndex > end_frame)
    {
        CELUX_DEBUG("Frame range exhausted: currentIndex={}, end_frame={}",
                    currentIndex, end_frame);
        throw py::stop_iteration(); // Stop iteration if range is exhausted
    }

    // py::gil_scoped_release release; // Uncomment if GIL management is needed
    CELUX_DEBUG("Reading next frame in iteration");
    torch::Tensor frame = readFrame();
    if (!frame.defined() || frame.numel() == 0)
    {
        CELUX_DEBUG("No more frames available for iteration");
        throw py::stop_iteration(); // Stop iteration if no more frames are available
    }
    CELUX_DEBUG("Frame retrieved successfully, currentIndex before increment: {}",
                currentIndex);
    currentIndex++;
    CELUX_DEBUG("currentIndex incremented to {}", currentIndex);
    CELUX_INFO("Returning frame number {}", currentIndex - 1);
    return frame;
}

void VideoReader::enter()
{
    CELUX_TRACE("enter() called: VideoReader entering context manager");
    // Resources are already initialized in the constructor
    CELUX_DEBUG("VideoReader is ready for use in context manager");
}

void VideoReader::exit(const py::object& exc_type, const py::object& exc_value,
                       const py::object& traceback)
{
    CELUX_TRACE("exit() called: VideoReader exiting context manager");
    close(); // Close the video reader and free resources
    CELUX_DEBUG("VideoReader resources have been cleaned up in context manager");
}

int VideoReader::length() const
{
    CELUX_TRACE("length() called: Returning totalFrames = {}", properties.totalFrames);
    return properties.totalFrames;
}
