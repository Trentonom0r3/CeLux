#include "MediaFormat.hpp"
#include "MediaDecoder.hpp"
#include <iostream>
#include <chrono>
#include <Timer.hpp>

void testVideoDecoding(FFmpeg::MediaFormat& mediaFormat, const std::string& codec) {
    try {
        Timer timer;
        mediaFormat.release();
        mediaFormat.open();

        FFmpeg::MediaDecoder decoder(mediaFormat);
        decoder.initializeVideoDecoder(mediaFormat.getVideoStreamIndex(), codec);
        decoder.initializeAudioDecoder(mediaFormat.getAudioStreamIndex());
        AVFrame* frame = av_frame_alloc();
        if (!frame) {
            throw std::runtime_error("Failed to allocate video frame.");
        }

        int frameCount = 0;
        while (decoder.decodeVideoFrame(frame)) {
            decoder.decodeAudioFrame(frame);
            frameCount++;
        }
        int fps = frameCount / timer.elapsed();

        std::cout << "Video decoding test passed!" << std::endl;
        std::cout << "Decoded " << frameCount << " frames in " << timer.elapsed() << " seconds (" << fps << " fps)" << ""
            << "Using codec: " << codec << std::endl;
        av_frame_free(&frame);

    }
    catch (const FFmpeg::Error::FFException& e) {
        std::cerr << "Video decoding test failed: " << e.what() << std::endl;
    }
}


void testAudioDecoding(const FFmpeg::MediaFormat& mediaFormat) {
    try {
        FFmpeg::MediaDecoder decoder(mediaFormat);
        decoder.initializeAudioDecoder(mediaFormat.getAudioStreamIndex());

        AVFrame* frame = av_frame_alloc();
        if (!frame) {
            throw std::runtime_error("Failed to allocate audio frame.");
        }

        int frameCount = 0;
        while (decoder.decodeAudioFrame(frame)) {
            std::cout << "Decoded audio frame: " << frameCount++ << std::endl;
        }

        std::cout << "Audio decoding test passed!" << std::endl;
        av_frame_free(&frame);
    }
    catch (const FFmpeg::Error::FFException& e) {
        std::cerr << "Audio decoding test failed: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Running MediaDecoder tests..." << std::endl;

    FFmpeg::MediaFormat mediaFormat("C:\\Users\\tjerf\\source\\repos\\FrameSmith\\Input.mp4");
    mediaFormat.open();

   //testVideoDecoding(mediaFormat, "vp9_cuvid");  // Attempt hardware-accelerated decoding for VP9
    testVideoDecoding(mediaFormat, "h264_cuvid"); // Attempt hardware-accelerated decoding for H264
 //   testAudioDecoding(mediaFormat);

    std::cout << "Tests completed." << std::endl;
    return 0;
}
