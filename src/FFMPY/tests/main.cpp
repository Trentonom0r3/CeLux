// main.cpp

#include "Decoder.hpp"
#include <NV12ToRGB.hpp>
#include <Timer.hpp>

using namespace ffmpy;
// Helper function to print all available filters
void printAvailableFilters()
{
    const AVFilter* filter = nullptr;
    void* opaque = nullptr; // Initialize iteration state

    std::cout << "Available FFmpeg Filters:" << std::endl;
    while ((filter = av_filter_iterate(&opaque)) != nullptr)
    {
        std::cout << " - " << filter->name;
        if (filter->description)
        {
            std::cout << ": " << filter->description;
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[])
{

    std::string videoFilePath =
        "C:\\Users\\tjerf\\source\\repos\\FrameSmith\\Input.mp4"; // Change to desired
                                                                  // video file path

    try
    {
        printAvailableFilters();
        // Create VideoDecoder instance with hardware acceleration
        Decoder decoder(videoFilePath, true,
                        "cuda"); // Change "cuda" to desired HW accel type

        // Get video properties
        Decoder::VideoProperties props = decoder.getVideoProperties();
        std::cout << "Video Properties:" << std::endl;
        std::cout << "Width: " << props.width << std::endl;
        std::cout << "Height: " << props.height << std::endl;
        std::cout << "FPS: " << props.fps << std::endl;
        std::cout << "Duration: " << props.duration << " seconds" << std::endl;
        std::cout << "Total Frames: " << props.totalFrames << std::endl;
        std::cout << "Pixel Format: " << av_get_pix_fmt_name(props.pixelFormat)
                  << std::endl;

        // Decode frames
        Frame frame;
        int frameCount = 0;
        Timer timer;
        torch::Tensor tnsr = torch::empty(
            {props.height, props.width, 3},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

        while (decoder.decodeNextFrame(frame))
        {
            ffmpy::conversion::NV12ToRGB<uint8_t>().convert(frame,
                                                            tnsr.data_ptr<uint8_t>());
            frameCount++;
        }
        double end = timer.elapsed();
        std::cout << "Decoding completed. Total frames decoded: " << frameCount
                  << std::endl;
        std::cout << "Time taken: " << end << " seconds" << std::endl;
        std::cout << "FPS: " << frameCount / end << std::endl;
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
