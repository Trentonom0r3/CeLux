// Python/VideoReader.cpp

#include "Python/VideoReader.hpp"
#include <pybind11/pybind11.h>
#include <torch/torch.h> // Ensure you have included the necessary Torch headers

namespace py = pybind11;
// Function to list all available FFmpeg filters
/*/**
 * Iterate over all registered filters.
 *
 * @param opaque a pointer where libavfilter will store the iteration state. Must
 *               point to NULL to start the iteration.
 *
 * @return the next registered filter or NULL when the iteration is
 *         finished
 
const AVFilter* av_filter_iterate(void** opaque);
*/ 

void list_ffmpeg_filters()
{
    // create void** opaque
    void* opaque = nullptr;
    const AVFilter* filter = nullptr;

    std::cout << "Available FFmpeg Filters:\n";
    std::cout << "--------------------------\n";

    while ((filter = av_filter_iterate(&opaque)))
    {
		const char* filter_name = filter->name;
		const char* filter_desc = filter->description;
		std::cout << "Filter Name: " << filter_name << "\n";
		std::cout << "Description: "
				  << (filter_desc ? filter_desc : "No description available") << "\n";
		std::cout << "---------------------------------------\n";
	}
}


VideoReader::VideoReader(const std::string& filePath, int numThreads,
                         const std::string& device, 
                         std::vector<std::tuple<std::string, std::string>> filters)
    : decoder(nullptr), currentIndex(0), start_frame(0), end_frame(-1)
{
    //set ffmpeg log level
    CELUX_INFO("VideoReader constructor called with filePath: {}", filePath);
    CELUX_INFO("Device: {}", device);
    if (numThreads > std::thread::hardware_concurrency())
    {
        throw std::invalid_argument("Number of threads cannot exceed hardware concurrency");
	}

    try
    {
        for (const auto& filter : filters)
        {
			CELUX_INFO("Filter: {}={}", std::get<0>(filter), std::get<1>(filter));
            filters_.push_back(std::make_shared<Filter>(std::get<0>(filter), std::get<1>(filter)));
		}

        torch::Device torchDevice = torch::Device(torch::kCPU);
        CELUX_INFO("Creating VideoReader instance");
        if (device == "cuda")
        {
            CELUX_INFO("Device is set to CUDA");
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
            CELUX_INFO("Device is set to CPU");
            CELUX_TRACE("Using CPU device for VideoReader");
            torchDevice = torch::Device(torch::kCPU);
        }
        else
        {
            CELUX_CRITICAL("Unsupported device: {}", device);
            throw std::invalid_argument("Unsupported device: " + device);
        }

        decoder =
            celux::Factory::createDecoder(torchDevice, filePath, numThreads, filters_);
        CELUX_INFO("Decoder created successfully");

        torch::Dtype torchDataType;

        torchDataType = findTypeFromBitDepth();

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
                     torch::TensorOptions().dtype(torchDataType).device(torchDevice))
                     .contiguous();

        CELUX_INFO("Torch tensor initialized with shape: [{}, {}, {}] :, "
                   "device: {}",
                   properties.height, properties.width, 3, device);
      //  list_ffmpeg_filters();
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
        CELUX_INFO("Adjusted start_frame to {}", start);
    }
    if (end < 0)
    {
        end = properties.totalFrames + end;
        CELUX_INFO("Adjusted end_frame to {}", end);
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
    CELUX_INFO("Adjusted end_frame to be exclusive: {}", end);

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

    CELUX_TRACE("Frame decoded successfully");
    return tensor;
}

void VideoReader::close()
{
    CELUX_INFO("Closing VideoReader");
    // Clean up decoder and other resources
    if (decoder)
    {
        CELUX_INFO("Closing decoder");
        decoder->close();
        decoder.reset();
        CELUX_INFO("Decoder closed and reset successfully");
    }
    else
    {
        CELUX_INFO("Decoder already closed or was never initialized");
    }
}

bool VideoReader::seek(double timestamp)
{
    CELUX_TRACE("Seeking to timestamp: {}", timestamp);
    bool success;

    // Release GIL during seeking
    {
        // py::gil_scoped_release release; // Uncomment if GIL management is needed
        success = decoder->seek(timestamp);
    }

    if (success)
    {
        CELUX_TRACE("Seek to timestamp {} successful", timestamp);
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
    CELUX_INFO("Number of supported decoders: {}", codecs.size());
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
    props["min_fps"] = properties.min_fps; // New property
    props["max_fps"] = properties.max_fps; // New property
    props["duration"] = properties.duration;
    props["total_frames"] = properties.totalFrames;
    props["pixel_format"] = av_get_pix_fmt_name(properties.pixelFormat)
                                ? av_get_pix_fmt_name(properties.pixelFormat)
                                : "Unknown";
    props["has_audio"] = properties.hasAudio;
    props["audio_bitrate"] = properties.audioBitrate;        // New property
    props["audio_channels"] = properties.audioChannels;      // New property
    props["audio_sample_rate"] = properties.audioSampleRate; // New property
    props["audio_codec"] = properties.audioCodec;            // New property
    props["bit_depth"] = properties.bitDepth;
    props["aspect_ratio"] = properties.aspectRatio; // New property
    props["codec"] = properties.codec;

    CELUX_INFO("Video properties retrieved and converted to Python dict");
    return props;
}

py::object VideoReader::operator[](const std::string& key) const
{
    CELUX_TRACE("__getitem__ called with key: {}", key);

    if (key == "width")
        return py::cast(properties.width); // Cast integer to py::object
    else if (key == "height")
        return py::cast(properties.height); // Cast integer to py::object
    else if (key == "fps")
        return py::cast(properties.fps); // Cast double to py::object
    else if (key == "duration")
        return py::cast(properties.duration); // Cast double to py::object
    else if (key == "total_frames")
        return py::cast(properties.totalFrames); // Cast integer to py::object
    else if (key == "pixel_format")
        return py::cast(
            av_get_pix_fmt_name(properties.pixelFormat)); // Cast string to py::object
    else if (key == "has_audio")
        return py::cast(properties.hasAudio); // Cast boolean to py::object
    else if (key == "audio_bitrate")
        return py::cast(properties.audioBitrate); // Cast integer to py::object
    else if (key == "audio_channels")
        return py::cast(properties.audioChannels); // Cast integer to py::object
    else if (key == "audio_sample_rate")
        return py::cast(properties.audioSampleRate); // Cast integer to py::object
    else if (key == "audio_codec")
        return py::cast(properties.audioCodec); // Cast string to py::object
    else if (key == "bit_depth")
        return py::cast(properties.bitDepth); // Cast integer to py::object
    else if (key == "aspect_ratio")
        return py::cast(properties.aspectRatio); // Cast double to py::object
    else if (key == "codec")
        return py::cast(properties.codec); // Cast string to py::object

    CELUX_WARN("Key '{}' not found in video properties", key);
    throw std::out_of_range("Key not found: " + key);
}


void VideoReader::reset()
{
    CELUX_INFO("Resetting VideoReader to the beginning");
    bool success = seek(0.0);
    if (success)
    {
        currentIndex = 0;
        CELUX_INFO("VideoReader reset successfully");
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

    // Calculate timestamp for the exact frame
    double exact_timestamp = static_cast<double>(frame_number) / properties.fps;
    CELUX_INFO("Exact timestamp for frame {}: {} seconds", frame_number,
               exact_timestamp);

    // Seek to the nearest keyframe before the target frame
    bool success = decoder->seekToNearestKeyframe(exact_timestamp);
    if (!success)
    {
        CELUX_WARN("Seeking to the nearest keyframe failed for frame number {}",
                   frame_number);
        return false;
    }

    // Decode frames until reaching the target frame number
    int current_frame = static_cast<int>(std::round(exact_timestamp * properties.fps));

    while (current_frame < frame_number)
    {
        CELUX_INFO("Within Seek: Decoding frame number {}", current_frame);
        readFrame();
        current_frame++;
    }

    CELUX_INFO("Seek to frame {} successful", frame_number);
    return true;
}

VideoReader& VideoReader::iter()
{
    CELUX_TRACE("iter() called: Preparing VideoReader for iteration");
    currentIndex = start_frame;
    bool success = seekToFrame(start_frame);
    for (int i = 0; i < start_frame; i++)
    {
		readFrame(); // Skip frames until start_frame
	}
    if (success)
    {
        CELUX_TRACE("VideoReader successfully seeked to start_frame: {}", start_frame);
    }
    else
    {
        CELUX_TRACE("Failed to seek to start_frame: {}", start_frame);
    }
    return *this;
}

torch::Tensor VideoReader::next()
{
    CELUX_TRACE("next() called: Retrieving next frame");
    if (end_frame >= 0 && currentIndex > end_frame)
    {
        CELUX_TRACE("Frame range exhausted: currentIndex={}, end_frame={}",
                    currentIndex, end_frame);
        throw py::stop_iteration(); // Stop iteration if range is exhausted
    }

    // py::gil_scoped_release release; // Uncomment if GIL management is needed
    CELUX_TRACE("Reading next frame in iteration");
    torch::Tensor frame = readFrame();
    if (!frame.defined() || frame.numel() == 0)
    {
        CELUX_TRACE("No more frames available for iteration");
        throw py::stop_iteration(); // Stop iteration if no more frames are available
    }
    CELUX_TRACE("Frame retrieved successfully, currentIndex before increment: {}",
                currentIndex);
    currentIndex++;
    CELUX_TRACE("currentIndex incremented to {}", currentIndex);
    CELUX_TRACE("Returning frame number {}", currentIndex - 1);
    return frame;
}

void VideoReader::enter()
{
    CELUX_TRACE("enter() called: VideoReader entering context manager");
    // Resources are already initialized in the constructor
    CELUX_INFO("VideoReader is ready for use in context manager");
}

void VideoReader::exit(const py::object& exc_type, const py::object& exc_value,
                       const py::object& traceback)
{
    CELUX_TRACE("exit() called: VideoReader exiting context manager");
    close(); // Close the video reader and free resources
    CELUX_INFO("VideoReader resources have been cleaned up in context manager");
}

int VideoReader::length() const
{
    CELUX_TRACE("length() called: Returning totalFrames = {}", properties.totalFrames);
    return properties.totalFrames;
}

torch::ScalarType VideoReader::findTypeFromBitDepth()
{
    int bit_depth = decoder->getBitDepth();
    CELUX_INFO("Bit depth of video: {}", bit_depth);
    torch::ScalarType torchDataType;
    switch (bit_depth)
    {
    case 8:
        CELUX_DEBUG("Setting tensor data type to torch::kUInt8");
        torchDataType = torch::kUInt8;
        break;
    case 10:
        CELUX_DEBUG("Setting tensor data type to torch::kUInt16");
        torchDataType = torch::kUInt16;
        break;
    case 16:
        CELUX_DEBUG("Setting tensor data type to torch::kUInt16");
        torchDataType = torch::kUInt16;
        break;
    case 32:
        CELUX_DEBUG("Setting tensor data type to torch::kUInt32");
        torchDataType = torch::kUInt32;
        break;
    default:
        CELUX_WARN("Unsupported bit depth: {}", bit_depth);
        throw std::runtime_error("Unsupported bit depth: " + std::to_string(bit_depth));
    }
    return torchDataType;
}
