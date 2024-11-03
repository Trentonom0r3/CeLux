// Python/VideoReader.cpp

#include "Python/VideoReader.hpp"
#include <pybind11/pybind11.h>
#include <torch/torch.h> // Ensure you have included the necessary Torch headers
#include "json.hpp" // Include the nlohmann/json header
using json = nlohmann::json;

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

extern "C"
{
#include <libavfilter/avfilter.h>
#include <libavutil/opt.h>
}

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
// Include C++ headers
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <iomanip> // For std::fixed and std::setprecision
// Include the nlohmann JSON library if you're loading a JSON file for required options
// #include <nlohmann/json.hpp>
// Helper function to convert AVOption type to string
std::string option_type_to_string(int type)
{
    CELUX_INFO("Starting option_type_to_string with type: {}", type);
    switch (type)
    {
    case AV_OPT_TYPE_FLAGS:
        return "Flags";
    case AV_OPT_TYPE_INT:
        return "Integer";
    case AV_OPT_TYPE_INT64:
        return "Integer64";
    case AV_OPT_TYPE_DOUBLE:
        return "Double";
    case AV_OPT_TYPE_FLOAT:
        return "Float";
    case AV_OPT_TYPE_STRING:
        return "String";
    case AV_OPT_TYPE_RATIONAL:
        return "Rational";
    case AV_OPT_TYPE_BINARY:
        return "Binary";
    case AV_OPT_TYPE_DICT:
        return "Dictionary";
    case AV_OPT_TYPE_UINT64:
        return "Unsigned Integer64";
    case AV_OPT_TYPE_CONST:
        return "Constant";
    case AV_OPT_TYPE_IMAGE_SIZE:
        return "Image Size";
    case AV_OPT_TYPE_PIXEL_FMT:
        return "Pixel Format";
    case AV_OPT_TYPE_SAMPLE_FMT:
        return "Sample Format";
    case AV_OPT_TYPE_VIDEO_RATE:
        return "Video Rate";
    case AV_OPT_TYPE_DURATION:
        return "Duration";
    case AV_OPT_TYPE_COLOR:
        return "Color";
    case AV_OPT_TYPE_BOOL:
        return "Boolean";
    case AV_OPT_TYPE_CHLAYOUT:
        return "Channel Layout";
    case AV_OPT_TYPE_FLAG_ARRAY:
        return "Flag Array";
    default:
        CELUX_WARN("Unknown AVOption type: {}", type);
        return "Unknown";
    }
}

// Helper function to convert AVOption flags to string
std::string option_flags_to_string(int flags)
{
    CELUX_INFO("Starting option_flags_to_string with flags: {}", flags);
    std::string result;
    if (flags & AV_OPT_FLAG_VIDEO_PARAM)
        result += "Video ";
    if (flags & AV_OPT_FLAG_AUDIO_PARAM)
        result += "Audio ";
    if (flags & AV_OPT_FLAG_FILTERING_PARAM)
        result += "Filtering ";
    if (flags & AV_OPT_FLAG_ENCODING_PARAM)
        result += "Encoding ";
    if (flags & AV_OPT_FLAG_DEPRECATED)
        result += "Deprecated ";
    if (flags & AV_OPT_FLAG_READONLY)
        result += "ReadOnly ";
    // Add more flags as needed
    return result.empty() ? "None" : result;
}

// Helper function to convert AVRational to string
std::string rational_to_string(const AVRational& r)
{
    CELUX_INFO("Starting rational_to_string with AVRational: {}/{}", r.num, r.den);
    std::ostringstream oss;
    if (r.den == 0)
    {
        oss << "N/A";
    }
    else
    {
        double value = static_cast<double>(r.num) / r.den;
        oss << value;
    }
    return oss.str();
}

// Helper function to extract default value as string
std::string get_option_default(const AVOption* option)
{
    CELUX_INFO("Starting get_option_default with option: {}", option->name);
    if (!option)
    {
        return "None";
    }

    std::ostringstream oss;

    // Determine if the option has a default value
    bool has_default = true;

    switch (option->type)
    {
    case AV_OPT_TYPE_STRING:
    case AV_OPT_TYPE_COLOR:
    case AV_OPT_TYPE_CHLAYOUT:
    case AV_OPT_TYPE_BINARY:
    case AV_OPT_TYPE_DICT:
    case AV_OPT_TYPE_IMAGE_SIZE:
        has_default =
            (option->default_val.str != nullptr && option->default_val.str[0] != '\0');
        if (has_default)
            oss << option->default_val.str;
        else
            oss << "No Default";
        break;

    case AV_OPT_TYPE_INT:
    case AV_OPT_TYPE_INT64:
    case AV_OPT_TYPE_UINT64:
    case AV_OPT_TYPE_FLAGS:
    case AV_OPT_TYPE_DURATION:
        oss << option->default_val.i64;
        break;

    case AV_OPT_TYPE_DOUBLE:
    case AV_OPT_TYPE_FLOAT:
        oss << std::fixed << std::setprecision(2) << option->default_val.dbl;
        break;

    case AV_OPT_TYPE_BOOL:
        oss << (option->default_val.i64 ? "true" : "false");
        break;

    case AV_OPT_TYPE_RATIONAL:
    case AV_OPT_TYPE_VIDEO_RATE:
        if (option->default_val.q.num != 0 || option->default_val.q.den != 0)
            oss << rational_to_string(option->default_val.q);
        else
            oss << "No Default";
        break;

    case AV_OPT_TYPE_SAMPLE_FMT:
    {
        AVSampleFormat fmt = static_cast<AVSampleFormat>(option->default_val.i64);
        if (fmt == AV_SAMPLE_FMT_NONE)
        {
            oss << "No Default";
        }
        else
        {
            const char* sampleName = av_get_sample_fmt_name(fmt);
            oss << (sampleName ? sampleName : "Unknown Format");
        }
    }
    break;

    case AV_OPT_TYPE_PIXEL_FMT:
    {
        AVPixelFormat fmt = static_cast<AVPixelFormat>(option->default_val.i64);
        if (fmt == AV_PIX_FMT_NONE)
        {
            oss << "No Default";
        }
        else
        {
            const char* name = av_get_pix_fmt_name(fmt);
            oss << (name ? name : "Unknown Format");
        }
    }
    break;

    case AV_OPT_TYPE_CONST:
        // Constants represent possible values, default is their value
        oss << option->default_val.i64;
        break;

    default:
        oss << "Unsupported Type";
        break;
    }

    return oss.str();
}


// Function to determine if an option is required based on the absence of a default
// value
bool is_option_required(const AVOption* opt)
{
    CELUX_INFO("Starting is_option_required with option: {}", opt->name);

    // Exclude deprecated options
    if (opt->flags & AV_OPT_FLAG_DEPRECATED)
        return false;

    switch (opt->type)
    {
    case AV_OPT_TYPE_STRING:
    case AV_OPT_TYPE_COLOR:
    case AV_OPT_TYPE_CHLAYOUT:
    case AV_OPT_TYPE_BINARY:
    case AV_OPT_TYPE_DICT:
    case AV_OPT_TYPE_IMAGE_SIZE:
        // Required if default string is nullptr or empty
        return (opt->default_val.str == nullptr || opt->default_val.str[0] == '\0');

    case AV_OPT_TYPE_INT:
    case AV_OPT_TYPE_INT64:
    case AV_OPT_TYPE_UINT64:
    case AV_OPT_TYPE_FLAGS:
        // Adjust based on whether a sentinel value indicates no default
        // For now, assume optional unless a special value indicates required
        return false;

    case AV_OPT_TYPE_DOUBLE:
    case AV_OPT_TYPE_FLOAT:
        // Zero might be a valid default; assume optional
        return false;

    case AV_OPT_TYPE_BOOL:
        // Booleans typically have a default value; assume optional
        return false;

    case AV_OPT_TYPE_RATIONAL:
    case AV_OPT_TYPE_VIDEO_RATE:
        // Required if numerator and denominator are zero
        return (opt->default_val.q.num == 0 && opt->default_val.q.den == 0);

    case AV_OPT_TYPE_DURATION:
        // Required if default is AV_NOPTS_VALUE
        return (opt->default_val.i64 == AV_NOPTS_VALUE);

    case AV_OPT_TYPE_PIXEL_FMT:
        // Required if default is AV_PIX_FMT_NONE
        return (opt->default_val.i64 == AV_PIX_FMT_NONE);

    case AV_OPT_TYPE_SAMPLE_FMT:
        // Required if default is AV_SAMPLE_FMT_NONE
        return (opt->default_val.i64 == AV_SAMPLE_FMT_NONE);

    case AV_OPT_TYPE_CONST:
    case AV_OPT_TYPE_FLAG_ARRAY:
        // Not user-settable or assume optional
        return false;

    default:
        // Unhandled types are assumed optional
        return false;
    }
}

void list_ffmpeg_filters(const std::string& output_filename)
{
    CELUX_INFO("Starting list_ffmpeg_filters with output file: {}", output_filename);

    // Create a JSON array to hold all filters
    json filters_json = json::array();

    // Create an opaque pointer for iteration
    void* opaque = nullptr;
    const AVFilter* filter = nullptr;

    CELUX_INFO("Starting to iterate over all filters");

    // Iterate over all available filters
    while ((filter = av_filter_iterate(&opaque)))
    {
        if (!filter)
        {
            CELUX_INFO("No more filters to iterate");
            break;
        }

        const char* filter_name = filter->name;
        const char* filter_desc = filter->description;
        CELUX_INFO("Processing filter: {}", filter_name);

        // Create a JSON object for this filter
        json filter_json;
        filter_json["filter_name"] = filter_name;
        filter_json["description"] =
            filter_desc ? filter_desc : "No description available";

        // Create an array to hold the options
        json options_json = json::array();

        // Check if the filter has a priv_class
        if (filter->priv_class)
        {
            CELUX_INFO("Filter '{}' has priv_class", filter_name);

            // Collect all options into a vector
            std::vector<const AVOption*> options;
            const AVOption* option = nullptr;
            while ((option = av_opt_next(&filter->priv_class, option)))
            {
                options.push_back(option);
            }

            // Build a mapping from unit names to constants (aliases)
            std::unordered_map<std::string, std::vector<const AVOption*>>
                unit_to_constants;
            for (const AVOption* opt : options)
            {
                if (opt->type == AV_OPT_TYPE_CONST && opt->unit)
                {
                    unit_to_constants[opt->unit].push_back(opt);
                }
            }

            // Process each option
            for (const AVOption* opt : options)
            {
                if (opt->type == AV_OPT_TYPE_CONST)
                {
                    // Skip constants here; they are used in possible_values
                    continue;
                }

                CELUX_INFO("Processing option '{}' for filter '{}'", opt->name,
                           filter_name);

                // Create a JSON object for this option
                json option_json;
                option_json["name"] = opt->name;
                option_json["help"] = opt->help ? opt->help : "";
                option_json["type"] = option_type_to_string(opt->type);
                option_json["default"] = get_option_default(opt);
                option_json["required"] = is_option_required(opt);
                option_json["flags"] = option_flags_to_string(opt->flags);
                option_json["flags_value"] = opt->flags;
                option_json["min"] = opt->min;
                option_json["max"] = opt->max;
                option_json["unit"] = opt->unit ? opt->unit : "";
                option_json["deprecated"] = (opt->flags & AV_OPT_FLAG_DEPRECATED) != 0;
                option_json["readonly"] = (opt->flags & AV_OPT_FLAG_READONLY) != 0;

                // Include possible values if available (aliases)
                if (opt->unit)
                {
                    auto it = unit_to_constants.find(opt->unit);
                    if (it != unit_to_constants.end())
                    {
                        const auto& constants = it->second;
                        json possible_values = json::array();
                        for (const AVOption* const_opt : constants)
                        {
                            json const_json;
                            const_json["name"] = const_opt->name;
                            const_json["value"] = get_option_default(const_opt);
                            const_json["help"] = const_opt->help ? const_opt->help : "";
                            possible_values.push_back(const_json);
                        }
                        option_json["possible_values"] = possible_values;
                    }
                }

                // Add the option to the options array
                options_json.push_back(option_json);
            }
        }
        else
        {
            CELUX_INFO("Filter '{}' does not have priv_class", filter_name);
        }

        // Add the options array to the filter JSON object
        filter_json["options"] = options_json;

        // Add the filter JSON object to the filters array
        filters_json.push_back(filter_json);
    }

    CELUX_INFO("Completed iterating over all filters");

    // Serialize the JSON object to a string
    std::string json_output = filters_json.dump(4); // 4-space indentation

    // Write the JSON string to the output file
    std::ofstream outfile(output_filename);
    if (!outfile.is_open())
    {
        CELUX_INFO("Failed to open file: {}", output_filename);
        return;
    }

    outfile << json_output;
    outfile.close();

    CELUX_INFO("Filter information has been written to {}", output_filename);
    std::cout << "Filter information has been written to " << output_filename
              << std::endl;
}



VideoReader::VideoReader(const std::string& filePath, int numThreads,
                         const std::string& device, 
                         std::vector<std::shared_ptr<FilterBase>> filters)
    : decoder(nullptr), currentIndex(0), start_frame(0), end_frame(-1), filters_(filters)
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
        CELUX_INFO("NUM FILTERS: {}", filters_.size());
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

        // Initialize tensor
        tensor = torch::empty(
                     {properties.height, properties.width, 3},
                     torch::TensorOptions().dtype(torchDataType).device(torchDevice))
                     .contiguous();

        CELUX_INFO("Torch tensor initialized with shape: [{}, {}, {}] :, "
                   "device: {}",
                   properties.height, properties.width, 3, device);
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
