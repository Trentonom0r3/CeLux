#include "Python/VideoReader.hpp"
#include <pybind11/pybind11.h>


namespace py = pybind11;

VideoReader::VideoReader(const std::string& filePath, const std::string& device,
                         const std::string& dataType, int buffer_size,
                         std::optional<torch::Stream> stream)
    : decoder(nullptr), currentIndex(0), start_frame(0), end_frame(-1),
      torchDevice(torch::Device(torch::kCPU)), // Initialize with CPU by default
      stopBuffering_(false), tensorBuffer_(nullptr)
{
    try
    {
        CELUX_INFO("Creating VideoReader");
        // Determine the backend and set torchDevice based on the device string
        celux::backend backend;
        if (device == "cuda")
        {
            if (!torch::cuda::is_available())
            {
                CELUX_CRITICAL("CUDA is not available. Please install a "
                							   "CUDA-enabled version of celux.");
                throw std::runtime_error("CUDA is not available. Please install a "
                                         "CUDA-enabled version of celux.");
            }
            if (torch::cuda::device_count() == 0)
            {
                CELUX_CRITICAL("No CUDA devices found. Please check your CUDA installation.");
                throw std::runtime_error(
                    "No CUDA devices found. Please check your CUDA installation.");
            }

            backend = celux::backend::CUDA;
            torchDevice = torch::Device(torch::kCUDA);
            CELUX_DEBUG("Using CUDA device for VideoReader");
            CELUX_TRACE("CUDA device count: {}", torch::cuda::device_count());
        }
        else if (device == "cpu")
        {
            CELUX_TRACE("Using CPU device for VideoReader");
            backend = celux::backend::CPU;
            torchDevice = torch::Device(torch::kCPU);
        }
        else
        {
            CELUX_CRITICAL("Unsupported device: {}" + device);
            throw std::invalid_argument("Unsupported device: " + device);
        }

        // Map dataType string to celux::dataType enum and torch::Dtype
        celux::dataType dtype;
        torch::Dtype torchDataType;

        if (dataType == "uint8")
        {
            torchDataType = torch::kUInt8;
            dtype = celux::dataType::UINT8;
        }
        else if (dataType == "float32")
        {
            torchDataType = torch::kFloat32;
            dtype = celux::dataType::FLOAT32;
        }
        else if (dataType == "float16")
        {
            torchDataType = torch::kFloat16;
            dtype = celux::dataType::FLOAT16;
        }
        else
        {
            throw std::invalid_argument("Unsupported dataType: " + dataType);
        }

        // Create the converter using the factory based on the backend and stream
        if (backend == celux::backend::CUDA)
        {
            if (stream.has_value())
            {
                CELUX_TRACE("Creating converter with provided CUDA stream");
                convert = celux::Factory::createConverter(
                    backend, celux::ConversionType::NV12ToRGB, dtype,
                   stream);
            }
            else
            {
                CELUX_TRACE("Creating converter with default CUDA stream");
                convert = celux::Factory::createConverter(
                    backend, celux::ConversionType::NV12ToRGB, dtype,
                    std::nullopt);
            }
        }
        else // CPU backend
        {
            CELUX_TRACE("Creating converter for CPU");
            // Assuming the converter for CPU does not require a CUDA stream
            convert = celux::Factory::createConverter(
                backend, celux::ConversionType::NV12ToRGB, dtype,
                std::nullopt); // Pass a default or dummy stream for CPU
        }

        // Create the decoder using the factory
        decoder = celux::Factory::createDecoder(backend, filePath, std::move(convert));

        // Retrieve video properties
        properties = decoder->getVideoProperties();

        // Initialize tensors based on backend and data type
        if (backend == celux::backend::CUDA)
        {
            // Initialize RGBTensor on CUDA device
            RGBTensor = torch::empty(
                {properties.height, properties.width, 3},
                torch::TensorOptions().dtype(torchDataType).device(torchDevice));

            // Initialize cpuTensor on CPU if needed for CPU operations
            cpuTensor = torch::empty(
                {properties.height, properties.width, 3},
                torch::TensorOptions().dtype(torchDataType).device(torchDevice));
        }
        else // CPU backend
        {
            // Initialize cpuTensor on CPU
            cpuTensor = torch::empty(
                {properties.height, properties.width, 3},
                torch::TensorOptions().dtype(torchDataType).device(torchDevice));

            // If RGBTensor is not used on CPU, consider not initializing it
            // Alternatively, initialize it on CPU if needed
            RGBTensor = torch::empty(
                {properties.height, properties.width, 3},
                torch::TensorOptions().dtype(torchDataType).device(torchDevice));
        }

        // Configure BufferConfig
        BufferConfig config;
        config.device = torchDevice;
        config.dtype = torchDataType;
        config.shape = {properties.height, properties.width, 3};
        config.queueSize = buffer_size;

        // Initialize tensorBuffer_
        tensorBuffer_ = std::make_unique<TensorRingBuffer>(config);

        // Start the background thread for buffering frames
        bufferThread_ = std::thread(&VideoReader::bufferFrames, this);
    }
    catch (const std::exception& ex)
    {
        CELUX_DEBUG("Exception in VideoReader constructor: ");
        throw; // Re-throw exception after logging
    }
}


VideoReader::~VideoReader()
{
    close();
}

void VideoReader::setRange(int start, int end)
{
    // Handle negative indices by converting them to positive frame numbers
    if (start < 0)
        start = properties.totalFrames + start;
    if (end < 0)
        end = properties.totalFrames + end;

    // Validate the adjusted frame range
    if (start < 0 || end < 0)
    {
        throw std::runtime_error("Frame indices out of range after adjustment.");
    }
    if (end <= start)
    {
        throw std::runtime_error(
            "end_frame must be greater than start_frame after adjustment.");
    }

    // Make end_frame exclusive by subtracting one
    end = end - 1;

    start_frame = start;
    end_frame = end;
}
void VideoReader::bufferFrames()
{
    while (!stopBuffering_)
    {
        bool success = tensorBuffer_->produce(
            [this](torch::Tensor& tensor)
            {
                int result;

                {
                    // Decode the next frame into 'tensor'
                    result = decoder->decodeNextFrame(tensor.data_ptr());
                }

                if (result == 1) // Frame decoded successfully
                {
                    return true;
                }
                else if (result == 0) // End of video stream
                {
                    stopBuffering_ = true;
                    tensorBuffer_->stop();
                    return false;
                }
                else // Decoding failed
                {
                    // Handle error as needed
                    stopBuffering_ = true;
                    tensorBuffer_->stop();
                    return false;
                }
            });

        if (!success || stopBuffering_)
            break;
    }
}

torch::Tensor VideoReader::readFrame()
{
    CELUX_TRACE("Reading frame at index: {}", currentIndex);
    if (tensorBuffer_->isStopped() && tensorBuffer_->size() == 0)
    {
        // No more frames available
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8));
    }

    torch::Tensor frame = tensorBuffer_->consume();
    if (!frame.defined() || frame.numel() == 0)
    {
        // No frame available
        // No more frames available
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8));
    }

    return frame;
}

void VideoReader::close()
{
    stopBuffering_ = true;
    tensorBuffer_->stop();

    if (bufferThread_.joinable())
    {
        CELUX_DEBUG("Joining buffer thread");
        bufferThread_.join();
    }

    if (convert) {
		CELUX_DEBUG("Closing converter");
        CELUX_DEBUG("Synchronizing Converter");
        convert->synchronize();
		convert.reset();
	}
    else {
		CELUX_DEBUG("Converter is null");
	}
    // Clean up decoder and other resources
    if (decoder)
    {
        CELUX_DEBUG("Closing decoder");
        decoder->close();
        decoder.reset();
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

torch::Tensor VideoReader::next()
{
    if (end_frame >= 0 && currentIndex > end_frame)
    {
        throw py::stop_iteration(); // Stop iteration if range is exhausted
    }
    py::gil_scoped_release release;

    torch::Tensor frame = readFrame();
    if (frame.numel() == 0)
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

int VideoReader::length() const
{
    return properties.totalFrames;
}

void VideoReader::sync()
{
    // Release GIL during synchronization
    {
       // py::gil_scoped_release release;
        if (convert)
        {
            convert->synchronize();
        }
    }
}
