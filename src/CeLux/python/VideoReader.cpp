// Python/VideoReader.cpp

#include "Python/VideoReader.hpp"
#include <pybind11/pybind11.h>
#include <torch/torch.h> // Ensure you have included the necessary Torch headers
#include <TensorBuilder.hpp>

namespace py = pybind11;
#define CHECK_TENSOR(tensor)                                                  \
	if (!tensor.defined() || tensor.numel() == 0)                              \
	{                                                                         \
		throw std::runtime_error("Invalid tensor: undefined or empty");        \
	}

VideoReader::VideoReader(const std::string& filePath, int numThreads,
                         std::vector<std::shared_ptr<FilterBase>> filters, std::string tensorShape)
    : decoder(nullptr), currentIndex(0), start_frame(0), end_frame(-1),
      start_time(-1.0), end_time(-1.0), filters_(filters)
{
    //set ffmpeg log level
    CELUX_INFO("VideoReader constructor called with filePath: {}", filePath);

    if (numThreads > std::thread::hardware_concurrency())
    {
        throw std::invalid_argument("Number of threads cannot exceed hardware concurrency");
	}

    try
    {

        torch::Device torchDevice = torch::Device(torch::kCPU);
        CELUX_INFO("Creating VideoReader instance");
       
        decoder =
            celux::Factory::createDecoder(torchDevice, filePath, numThreads, filters_);
        CELUX_INFO("Decoder created successfully");

        torch::Dtype torchDataType;

        torchDataType = findTypeFromBitDepth();

        // Retrieve video properties
        properties = decoder->getVideoProperties();
        
        for (auto& filter : filters_)
        { // Iterate through filters_
            // Use dynamic_cast to check if the filter is of type Scale
            if (Scale* scaleFilter = dynamic_cast<Scale*>(filter.get()))
            {
                properties.width = std::stoi(scaleFilter->getWidth());
                properties.height = std::stoi(scaleFilter->getHeight());
            }
        }
    
        CELUX_INFO("Video properties retrieved: width={}, height={}, fps={}, "
                   "duration={}, totalFrames={}, pixelFormat={}, hasAudio={}",
                   properties.width, properties.height, properties.fps,
                   properties.duration, properties.totalFrames,
                   av_get_pix_fmt_name(properties.pixelFormat), properties.hasAudio);

        TensorBuilder builder(tensorShape);
        builder.createTensor(properties.height, properties.width, torchDataType,
                             torchDevice);
        // Initialize tensor
        tensor = builder.getTensor().contiguous().clone();
        CHECK_TENSOR(tensor);
        
  //  list_ffmpeg_filters("ffmpeg_filters.json");
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

void VideoReader::setRange(std::variant<int, double> start,
                           std::variant<int, double> end)
{
    // Check if both start and end are of the same type
    if (start.index() != end.index())
    {

        throw std::invalid_argument("Start and end must be of the same type.");
    }

    // Set the range based on the type of start and end
    if (std::holds_alternative<int>(start) && std::holds_alternative<int>(end))
    {
        int startFrame = std::get<int>(start);
        int endFrame = std::get<int>(end);
        setRangeByFrames(startFrame, endFrame);
    }
    else if (std::holds_alternative<double>(start) &&
             std::holds_alternative<double>(end))
    {
        double startTime = std::get<double>(start);
        double endTime = std::get<double>(end);
        setRangeByTimestamps(startTime, endTime);
    }
    else
    {

        throw std::invalid_argument("Unsupported type for start and end.");
    }
}

void VideoReader::setRangeByFrames(int startFrame, int endFrame)
{
    CELUX_INFO("Setting frame range: start={}, end={}", startFrame, endFrame);

    // Handle negative indices by converting them to positive frame numbers
    if (startFrame < 0)
    {
        startFrame = properties.totalFrames + startFrame;
        CELUX_INFO("Adjusted start_frame to {}", startFrame);
    }
    if (endFrame < 0)
    {
        endFrame = properties.totalFrames + endFrame;
        CELUX_INFO("Adjusted end_frame to {}", endFrame);
    }

    // Validate the adjusted frame range
    if (startFrame < 0 || endFrame < 0)
    {
        CELUX_ERROR("Frame indices out of range after adjustment: start={}, end={}",
                    startFrame, endFrame);
        throw std::runtime_error("Frame indices out of range after adjustment.");
    }
    if (endFrame <= startFrame)
    {
        CELUX_ERROR("Invalid frame range: end_frame ({}) must be greater than "
                    "start_frame ({}) after adjustment.",
                    endFrame, startFrame);
        throw std::runtime_error(
            "end_frame must be greater than start_frame after adjustment.");
    }

    // Make end_frame exclusive by subtracting one
    endFrame = endFrame - 1;
    CELUX_INFO("Adjusted end_frame to be exclusive: {}", endFrame);

    start_frame = startFrame;
    end_frame = endFrame;
    CELUX_INFO("Frame range set: start_frame={}, end_frame={}", start_frame, end_frame);
}

void VideoReader::setRangeByTimestamps(double startTime, double endTime)
{
    CELUX_INFO("Setting timestamp range: start={}, end={}", startTime, endTime);

    // Validate the timestamp range
    if (startTime < 0 || endTime < 0)
    {
        CELUX_ERROR("Timestamps cannot be negative: start={}, end={}", startTime,
                    endTime);
        throw std::invalid_argument("Timestamps cannot be negative.");
    }
    if (endTime <= startTime)
    {
        CELUX_ERROR("Invalid timestamp range: end ({}) must be greater than start ({})",
                    endTime, startTime);
        throw std::invalid_argument("end must be greater than start.");
    }

    // Set the timestamp range
    start_time = startTime;
    end_time = endTime;
    CELUX_INFO("Timestamp range set: start_time={}, end_time={}", start_time, end_time);
}

torch::Tensor VideoReader::readFrame()
{
    CELUX_TRACE("readFrame() called");
    py::gil_scoped_release release; // Release GIL before calling decoder

    double frame_timestamp = 0.0;
    bool success = decoder->decodeNextFrame(tensor.data_ptr(), &frame_timestamp);
    if (!success)
    {
        CELUX_WARN("Decoding failed or no more frames available");
        return torch::Tensor(); // Return an empty tensor if decoding failed
    }

    // Update current timestamp
    current_timestamp = frame_timestamp;

    CELUX_TRACE("Frame decoded successfully at timestamp: {}", current_timestamp);
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

    // Reset iterator state
    currentIndex = 0;
    current_timestamp = 0.0;
    bufferedFrame = torch::Tensor(); // Clear any old buffered frame
    hasBufferedFrame = false;

    if (start_time >= 0.0 && end_time > 0.0)
    {
        // Using timestamp range
        CELUX_INFO("Using timestamp range for iteration: start_time={}, end_time={}",
                   start_time, end_time);

        bool success = seek(start_time);
        if (!success)
        {
            CELUX_ERROR("Failed to seek to start_time: {}", start_time);
            throw std::runtime_error("Failed to seek to start_time.");
        }

        // -------------------------------------------------------
        // 1) DECODING + DISCARD loop
        // -------------------------------------------------------
        // Keep reading frames, discarding them, until we hit >= start_time.
        while (true)
        {
            // Attempt to decode a frame
            torch::Tensor f = readFrame();
            if (!f.defined() || f.numel() == 0)
            {
                // No more frames, or decode error
                CELUX_WARN("Ran out of frames while discarding up to start_time={}",
                           start_time);
                break;
            }

            // current_timestamp was updated in readFrame().
            if (current_timestamp >= start_time)
            {
                // We have reached or passed start_time
                // --> store this frame for later return in next()
                bufferedFrame = f;
                hasBufferedFrame = true;
                CELUX_DEBUG("Discard loop found first frame at timestamp {}",
                            current_timestamp);
                break;
            }
            // else discard and loop again
        }
        // -------------------------------------------------------

        current_timestamp = std::max(current_timestamp, start_time);
    }
    else if (start_frame >= 0 && end_frame >= 0)
    {
        // Using frame range
        CELUX_INFO("Using frame range for iteration: start_frame={}, end_frame={}",
                   start_frame, end_frame);
        bool success = seekToFrame(start_frame);
        if (!success)
        {
            CELUX_ERROR("Failed to seek to start_frame: {}", start_frame);
            throw std::runtime_error("Failed to seek to start_frame.");
        }
        currentIndex = start_frame;
        current_timestamp = static_cast<double>(currentIndex) / properties.fps;
    }
    else
    {
        // No range set; start from the beginning
        CELUX_INFO("No range set; starting from the beginning");
        bool success = seek(0.0);
        if (!success)
        {
            CELUX_ERROR("Failed to seek to the beginning of the video");
            throw std::runtime_error("Failed to seek to the beginning of the video.");
        }
        current_timestamp = 0.0;
    }

    // Return self for iterator protocol
    return *this;
}

torch::Tensor VideoReader::next()
{
    CELUX_TRACE("next() called: Retrieving next frame");

    // If we have a buffered frame from the discard loop, consume it first.
    torch::Tensor frame;
    if (hasBufferedFrame)
    {
        frame = bufferedFrame;
        hasBufferedFrame = false;
        // current_timestamp is already set by readFrame() earlier.
    }
    else
    {
        // Otherwise decode the next frame
        frame = readFrame();
        if (!frame.defined() || frame.numel() == 0)
        {
            CELUX_INFO("No more frames available (decode returned empty).");
            throw py::stop_iteration();
        }
    }

    // -- Now check if we exceeded the time range AFTER decoding.
    if (start_time >= 0.0 && end_time > 0.0)
    {
        // If the current frame's timestamp is >= end_time, skip/stop. end time + 1 frame
        if (current_timestamp > end_time + 1/properties.fps)
        {
            CELUX_DEBUG("Frame timestamp {} >= end_time {}, skipping frame.",
                        current_timestamp, end_time);
            throw py::stop_iteration();
        }
    }
    else if (start_frame >= 0 && end_frame >= 0)
    {
        if (currentIndex > end_frame)
        {
            CELUX_DEBUG("Frame range exhausted: currentIndex={}, end_frame={}",
                        currentIndex, end_frame);
            throw py::stop_iteration();
        }
    }

    currentIndex++;
    CELUX_TRACE("Returning frame index={}, timestamp={}", currentIndex - 1,
                current_timestamp);
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
    case 12:
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
