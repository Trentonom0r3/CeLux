
#include "Python/VideoWriter.hpp"
#include <torch/extension.h>

VideoWriter::VideoWriter(const std::string& filePath, int width, int height, float fps,
                         bool as_numpy, std::string dtype)
    : encoder(nullptr), as_numpy(as_numpy)
{
    try
    {
        torch::Dtype torchDataType;
        py::dtype npDataType;

        if (dtype == "uint8")
        {
            torchDataType = torch::kUInt8;
            npDataType = py::dtype::of<uint8_t>();
            convert =
                std::make_unique<ffmpy::conversion::gpu::cuda::RGBToNV12<uint8_t>>();
        }
        else if (dtype == "float32")
        {
            torchDataType = torch::kFloat32;
            npDataType = py::dtype::of<float>();
            convert =
                std::make_unique<ffmpy::conversion::gpu::cuda::RGBToNV12<float>>();
        }
        else if (dtype == "float16")
        {
            torchDataType = torch::kFloat16;
            npDataType = py::dtype("float16");
            convert =
                std::make_unique<ffmpy::conversion::gpu::cuda::RGBToNV12<__half>>();
        }
        else
        {
            throw std::invalid_argument("Unsupported dataType: " + dtype);
        }

        bool useHardware = true;
        std::string hwType = "cuda";
        ffmpy::Encoder::VideoProperties props;
        props.width = width;
        props.height = height;
        props.fps = fps;
        props.pixelFormat = AV_PIX_FMT_CUDA;
        props.codecName = "h264_nvenc";

        encoder = std::make_unique<ffmpy::Encoder>(filePath, props, useHardware, hwType,
                                                   std::move(convert));

        cudaMalloc(&npBuffer, width * height * 3);
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
    cudaFree(npBuffer);
}

bool VideoWriter::writeFrame(py::object frame)
{
    try
    {
        if (py::isinstance<py::array>(frame))
        {
            // Handle NumPy array case
            py::array npFrame = py::cast<py::array>(frame);
            void* src = npFrame.mutable_data();
            size_t size = npFrame.nbytes();

            // Copy data from CPU (NumPy) to GPU (npBuffer)
            copyTo(src, npBuffer, size);
            encoder->encodeFrame(npBuffer);
        }
        else
        {
            // Handle Torch tensor case
            torch::Tensor tensorFrame = py::cast<torch::Tensor>(frame);

            // Ensure tensor is contiguous for correct memory layout
            if (!tensorFrame.is_contiguous())
            {
                tensorFrame = tensorFrame.contiguous();
            }

            // Handle both CPU and GPU tensors appropriately
            if (tensorFrame.device().is_cpu())
            {
                // Copy from CPU tensor to GPU (npBuffer)
                copyTo(tensorFrame.data_ptr(), npBuffer, tensorFrame.nbytes());
            }
            else if (tensorFrame.device().is_cuda())
            {
                // If tensor is already on GPU, just get the pointer
                encoder->encodeFrame(tensorFrame.data_ptr());
                return true;
            }

            encoder->encodeFrame(npBuffer);
            return true;
        }
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

void VideoWriter::copyTo(void* src, void* dst, size_t size)
{
    cudaError_t err;
    err = cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Error copying data to host: " +
                                 std::string(cudaGetErrorString(err)));
    }
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
