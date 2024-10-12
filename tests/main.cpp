// main.cpp

#include "Decoder.hpp"
#include "Frame.hpp"
#include <iostream>
#include <stdexcept>
#include <Timer.hpp>
#include <Encoder.hpp>
#include <RGBToNV12.hpp>


using namespace ffmpy;

int main()
{//"C:\Users\tjerf\source\repos\FrameSmith\Input.mp4"
    std::string inputFilePath =
        "C:\\Users\\tjerf\\source\\repos\\FrameSmith\\Input.mp4"; // Update with your input file
    std::string outputFilePath =
        "C:\\Users\\tjerf\\source\\repos\\FrameSmith\\Output.mp4"; // Update with your output file
    ffmpy::Encoder::VideoProperties props;
    props.width = 1920;
    props.height = 1080;
    props.fps = 30.0;
    props.pixelFormat = AV_PIX_FMT_CUDA; // Or any other supported format
    props.codecName = "h264_nvenc";      // Ensure this encoder is supported

    std::unique_ptr<ffmpy::conversion::IConverter> converter =
        std::make_unique<ffmpy::conversion::RGBToNV12<uint8_t>>();

    ffmpy::Encoder encoder(outputFilePath, props, true, "cuda", std::move(converter));

    try
    {
        std::unique_ptr<ffmpy::conversion::IConverter> converter =
            std::make_unique<ffmpy::conversion::NV12ToRGB<uint8_t>>();

        // Initialize the TestDecoder with hardware acceleration
        Decoder TestDecoder(inputFilePath, true, "cuda", std::move(converter));

        // Retrieve video properties
        Decoder::VideoProperties props = TestDecoder.getVideoProperties();
        std::cout << "Video Properties:" << std::endl;
        std::cout << "Width: " << props.width << std::endl;
        std::cout << "Height: " << props.height << std::endl;
        std::cout << "FPS: " << props.fps << std::endl;
        std::cout << "Duration: " << props.duration << " seconds" << std::endl;
        std::cout << "Total Frames: " << props.totalFrames << std::endl;
        std::cout << "Pixel Format: " << av_get_pix_fmt_name(props.pixelFormat)
                  << std::endl;

        void* data = nullptr;
        cudaMalloc(&data, props.width * props.height * 3);
        int frameCount = 0;
        Timer timer;
        while (TestDecoder.decodeNextFrame(data)) //decoder handles conversion internally
        {
            encoder.encodeFrame(data);
            frameCount++;
        }

        double duration = timer.elapsed();
        // Finalize encoding

        std::cout << "Decoding and encoding completed. Total frames processed: "
                  << frameCount << std::endl;
        std::cout << "Total time taken: " << duration << " s" << std::endl;
        std::cout << "FPS : " << frameCount / duration << std::endl;
    }
    catch (const ffmpy::error::FFException& ex)
    {
        std::cerr << "FFmpeg Exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Standard Exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
