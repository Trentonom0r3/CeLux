
#include "Python/VideoWriter.hpp"
#include <Factory.hpp>
#include <torch/extension.h>

VideoWriter::VideoWriter(const std::string& filePath, int width, int height, float fps,
                         const std::string& device, const std::string& dataType)
    : encoder(nullptr)
{
    try
    {
        std::cout << "Creating VideoWriter\n" << std::endl;
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

            props.codecName = "h264";
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
        std::cout << "Creating encoder\n" << std::endl;
        // Create the converter using the factory
        convert = celux::Factory::createConverter(
            backend, celux::ConversionType::RGBToNV12, dtype);
        std::cout << "Converter created\n" << std::endl;

      encoder = celux::Factory::createEncoder(backend, filePath, props, std::move(convert));
      std::cout << "Encoder created\n" << std::endl;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Exception in VideoReader constructor: " << ex.what() << std::endl;
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
        std::cout << "Writing frame\n" << std::endl;
        // If tensor is already on GPU, just get the pointer
        encoder->encodeFrame(tensorFrame.data_ptr());
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Exception in writeFrame: " << ex.what() << std::endl;
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
    if (encoder)
    {
        encoder->close();
    }
}
