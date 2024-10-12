
#include "Python/VideoWriter.hpp"
#include <torch/extension.h>

VideoWriter::VideoWriter(const std::string& filePath, py::dict config)
	: encoder(nullptr), as_numpy(as_numpy)
{
    try
    {
        torch::Dtype torchDataType;
        py::dtype npDataType;
        std::string dataType = config["dtype"].cast<std::string>();

        if (dataType == "uint8")
        {
            torchDataType = torch::kUInt8;
            npDataType = py::dtype::of<uint8_t>();
            convert = std::make_unique<ffmpy::conversion::RGBToNV12<uint8_t>>();
        }
        else if (dataType == "float32")
        {
            torchDataType = torch::kFloat32;
            npDataType = py::dtype::of<float>();
            convert = std::make_unique<ffmpy::conversion::RGBToNV12<float>>();
        }
        else if (dataType == "float16")
        {
            torchDataType = torch::kFloat16;
            npDataType = py::dtype("float16");
            convert = std::make_unique<ffmpy::conversion::RGBToNV12<__half>>();
        }
        else
        {
            throw std::invalid_argument("Unsupported dataType: " + dataType);
        }

        bool useHardware = true;
        std::string hwType = "cuda";
        ffmpy::Encoder::VideoProperties props;
        props.width = config["width"].cast<int>();
        props.height = config["height"].cast<int>();
        props.fps = config["fps"].cast<double>();
        props.pixelFormat = AV_PIX_FMT_CUDA;
        props.codecName = "h264_nvenc";

        encoder = std::make_unique<ffmpy::Encoder>(filePath, props, useHardware, hwType,
                                                   std::move(convert));

        //npBuffer
        cudaMalloc(&npBuffer, props.width * props.height * 3);
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
    if (py::isinstance<py::array>(frame))
    {
        copyTo(frame.ptr(), npBuffer, py::cast<py::array>(frame).nbytes(), CopyType::HOST);
        encoder->encodeFrame(npBuffer);
    }
    else if (py::isinstance<torch::Tensor>(frame))
    {
        auto tensor = py::cast<torch::Tensor>(frame);
		encoder->encodeFrame(tensor.data_ptr());
	}
    else
    {
		throw std::invalid_argument("Unsupported frame type: " + std::string(py::str(frame)));
    }
}

std::vector<std::string> VideoWriter::supportedCodecs()
{
    return encoder->listSupportedEncoders();
}

void VideoWriter::copyTo(void* src, void* dst, size_t size, CopyType type)
{
    cudaError_t err;
    err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, convert->getStream());
    if (err != cudaSuccess)
    {
		throw std::runtime_error("Error copying data to host: " + std::string(cudaGetErrorString(err)));
	}
}

void VideoWriter::close()
{
    if (convert)
    {
		convert->synchronize();
		convert.reset();
	}
    if (encoder)
    {
		encoder->close();
		encoder.reset();
	}
}
