
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
        if (device == "cuda")
        {
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
        CELUX_DEBUG("Writing frame\n");
        // If tensor is already on GPU, just get the pointer
        encoder->encodeFrame(tensorFrame.data_ptr());
    }
    catch (const std::exception& ex)
    {
        CELUX_DEBUG("Exception in writeFrame: ");
        throw; // Re-throw exception after logging
    }
}

std::vector<std::string> VideoWriter::supportedCodecs()
{
    return encoder->listSupportedEncoders();
}

void VideoWriter::close()
{
    if (convert)
    {
        convert->synchronize();
    }
}
