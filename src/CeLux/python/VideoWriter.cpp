
#include "Python/VideoWriter.hpp"
#include <Factory.hpp>
#include <torch/extension.h>

VideoWriter::VideoWriter(const std::string& filePath, int width, int height, float fps,
                         const std::string& device, const std::string& dataType, std::optional<torch::Stream> stream)
    : encoder(nullptr)
{
    try
    {
        CELUX_DEBUG("Creating VideoWriter\n");
        celux::Encoder::VideoProperties props;
        props.width = width;
        props.height = height;
        props.fps = fps;
        props.pixelFormat = AV_PIX_FMT_NV12;
        // Determine the backend enum from the device string
        celux::backend backend;
        torch::DeviceType deviceType;
        if (device == "cuda")
        {
            deviceType = torch::kCUDA;
            props.codecName = "h264_nvenc";
            if (!torch::cuda::is_available())
            {
                throw std::runtime_error("CUDA is not available. Please install a "
                                         "CUDA-enabled version of celux.");
            }
            if (torch::cuda::device_count() == 0)
            {
                throw std::runtime_error(
                    "No CUDA devices found. Please check your CUDA installation.");
            }

            backend = celux::backend::CUDA;
        }
        else if (device == "cpu")
        {

            deviceType = torch::kCPU;
            props.codecName = "libx264";
            backend = celux::backend::CPU;
        }
        else
        {
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
        CELUX_DEBUG("Creating encoder\n");
        // Create the converter using the factory
        if (!stream.has_value())
        {
            convert = celux::Factory::createConverter(
				backend, celux::ConversionType::RGBToNV12, dtype, std::nullopt);
		}
        else
        {
            convert = celux::Factory::createConverter(
				backend, celux::ConversionType::RGBToNV12, dtype, stream.value());
		}
        CELUX_DEBUG("Converter created\n");
        intermediateFrame = torch::empty({height, width, 3}, torch::TensorOptions().dtype(torchDataType).device(deviceType));
         encoder = celux::Factory::createEncoder(backend, filePath, props, std::move(convert));
      CELUX_DEBUG("Encoder created\n");
    }
    catch (const std::exception& ex)
    {
        CELUX_DEBUG("Exception in VideoReader constructor: ");
        throw; // Re-throw exception after logging
    }
}

VideoWriter::~VideoWriter()
{
    close();
    // cudaFree(npBuffer);
}

bool VideoWriter::writeFrame(torch::Tensor tensorFrame)
{
    try
    {
        CELUX_DEBUG("Writing frame");

        // Validate input tensor
        if (!tensorFrame.is_contiguous())
        {
            CELUX_WARN("Input tensor is not contiguous. Making it contiguous.");
            tensorFrame = tensorFrame.contiguous();
        }

        // Ensure tensor is on the correct device
        if (tensorFrame.device() != intermediateFrame.device())
        {
            CELUX_DEBUG("Input tensor is on a different device. Moving it to the "
                        "encoder's device.");
            tensorFrame = tensorFrame.to(intermediateFrame.device());
        }

        // Ensure data types match
        if (tensorFrame.dtype() != intermediateFrame.dtype())
        {
            CELUX_DEBUG("Input tensor dtype does not match intermediate frame. Casting "
                        "tensor.");
            tensorFrame = tensorFrame.to(intermediateFrame.dtype());
        }

        // Copy data into the intermediate frame
        intermediateFrame.copy_(tensorFrame, /*non_blocking=*/true);
        CELUX_DEBUG("Frame data copied to intermediateFrame");

        // Encode the frame
        bool success = encoder->encodeFrame(intermediateFrame.data_ptr());

        if (!success)
        {
            CELUX_ERROR("Failed to encode frame");
            return false;
        }

        CELUX_DEBUG("Frame encoded successfully");
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
    return encoder->listSupportedEncoders();
}

void VideoWriter::close()
{
    try
    {
        CELUX_DEBUG("Closing VideoWriter");

        if (encoder)
        {
            CELUX_DEBUG("Finalizing encoder");
            encoder->finalize(); // Ensure all buffered frames are processed
            encoder.reset();     // Release the encoder
            CELUX_DEBUG("Encoder finalized and reset");
        }

        if (intermediateFrame.defined())
        {
            intermediateFrame = torch::Tensor(); // Release the tensor
            CELUX_DEBUG("Intermediate frame tensor released");
        }

        CELUX_DEBUG("VideoWriter closed successfully");
    }
    catch (const std::exception& ex)
    {
        CELUX_ERROR("Exception in close(): {}", ex.what());
        // Handle or re-throw as needed
    }
}
