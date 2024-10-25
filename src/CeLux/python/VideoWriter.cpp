// Python/VideoWriter.cpp

#include "Python/VideoWriter.hpp"
#include <Factory.hpp>
#include <torch/extension.h>

VideoWriter::VideoWriter(const std::string& filePath, int width, int height, float fps,
                         const std::string& device,
                         std::optional<torch::Stream> stream)
    : encoder(nullptr)
{
    try
    {
        CELUX_INFO("Initializing VideoWriter");
        CELUX_DEBUG("Creating VideoWriter with parameters - FilePath: {}, Width: {}, "
                    "Height: {}, FPS: {}, Device: {}",
                    filePath, width, height, fps, device);

        // Set up video properties
        celux::Encoder::VideoProperties props;
        props.width = width;
        props.height = height;
        props.fps = fps;
        props.pixelFormat = AV_PIX_FMT_YUV420P;
        props.bitDepth = 8;
        CELUX_DEBUG(
            "Video properties set - Width: {}, Height: {}, FPS: {}, PixelFormat: {}",
            props.width, props.height, props.fps,
            av_get_pix_fmt_name(props.pixelFormat));

        if (device == "cuda")
        {
            deviceType = torch::kCUDA;
            props.codecName = "h264_nvenc";
            CELUX_DEBUG("Device set to CUDA");

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
            CELUX_INFO("Using CUDA backend for VideoWriter");
        }
        else if (device == "cpu")
        {
            deviceType = torch::kCPU;
            props.codecName = "libx264";
            CELUX_INFO("Using CPU backend for VideoWriter");
        }
        else
        {
            CELUX_ERROR("Unsupported device specified: {}", device);
            throw std::invalid_argument("Unsupported device: " + device);
        }

        torch::Dtype torchDataType = torch::kUInt8;

        CELUX_DEBUG("Creating converter using factory");
        // Create the converter using the factory
        if (!stream.has_value())
        {
            convert = celux::Factory::createConverter(deviceType, 8,
                                                      AV_PIX_FMT_RGB24,
                                                  std::nullopt);
            CELUX_DEBUG("Converter created without custom stream");
        }
        else
        {
            convert = celux::Factory::createConverter(deviceType, 8,
                                                      AV_PIX_FMT_RGB24, std::nullopt);
            CELUX_DEBUG("Converter created with provided CUDA stream");
        }

        encoder = celux::Factory::createEncoder(deviceType, filePath, props,
                                                std::move(convert));
        CELUX_INFO("Encoder created successfully with codec: {}", props.codecName);
    }
    catch (const std::exception& ex)
    {
        CELUX_ERROR("Exception in VideoWriter constructor: {}", ex.what());
        throw; // Re-throw exception after logging
    }
}

VideoWriter::~VideoWriter()
{
    CELUX_INFO("Destroying VideoWriter");
    close();
    // cudaFree(npBuffer); // Uncomment if needed
    CELUX_INFO("VideoWriter destroyed successfully");
}

bool VideoWriter::writeFrame(torch::Tensor tensorFrame)
{
    try
    {
        CELUX_TRACE("writeFrame() called");
        CELUX_DEBUG("Writing frame");
        if (deviceType != tensorFrame.device().type())
		{
			CELUX_ERROR("Input tensor device type does not match VideoWriter device type");
			throw std::invalid_argument("Input tensor device type does not match VideoWriter device type");
		}
        // Validate input tensor
        if (!tensorFrame.is_contiguous())
        {
            CELUX_WARN("Input tensor is not contiguous. Making it contiguous.");
            tensorFrame = tensorFrame.contiguous();
            CELUX_INFO("Input tensor made contiguous");
        }

        // Encode the frame
        CELUX_DEBUG("Encoding the frame using encoder");
        bool success = encoder->encodeFrame(tensorFrame.data_ptr());

        if (!success)
        {
            CELUX_ERROR("Failed to encode frame");
            return false;
        }

        CELUX_INFO("Frame encoded successfully");
        return true;
    }
    catch (const std::exception& ex)
    {
        CELUX_ERROR("Exception in writeFrame: {}", ex.what());
        return false; // Indicate failure
    }
}

std::vector<std::string> VideoWriter::supportedCodecs()
{
    CELUX_TRACE("supportedCodecs() called");
    std::vector<std::string> codecs = encoder->listSupportedEncoders();
    CELUX_DEBUG("Number of supported encoders: {}", codecs.size());
    for (const auto& codec : codecs)
    {
        CELUX_TRACE("Supported encoder: {}", codec);
    }
    return codecs;
}

void VideoWriter::close()
{
    CELUX_INFO("Closing VideoWriter");
    try
    {
        if (encoder)
        {
            CELUX_DEBUG("Finalizing encoder");
            bool finalized =
                encoder->finalize(); // Ensure all buffered frames are processed
            if (finalized)
            {
                CELUX_INFO("Encoder finalized successfully");
            }
            else
            {
                CELUX_WARN("Encoder finalization failed or was incomplete");
            }
            encoder.reset(); // Release the encoder
            CELUX_DEBUG("Encoder reset successfully");
        }
        else
        {
            CELUX_WARN("Encoder was not initialized or already reset");
        }

        CELUX_INFO("VideoWriter closed successfully");
    }
    catch (const std::exception& ex)
    {
        CELUX_ERROR("Exception in close(): {}", ex.what());
    }
}