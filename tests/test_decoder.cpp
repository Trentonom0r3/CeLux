#include "MediaFormat.hpp"
#include "MediaDecoder.hpp"
#include <iostream>

void testVideoDecoding(const FFmpeg::MediaFormat& mediaFormat) {
    try {
        FFmpeg::MediaDecoder decoder(mediaFormat);
        decoder.initializeVideoDecoder(mediaFormat.getVideoStreamIndex());

        AVFrame* frame = av_frame_alloc();
        if (!frame) {
            throw std::runtime_error("Failed to allocate video frame.");
        }

        int frameCount = 0;
        while (decoder.decodeVideoFrame(frame)) {
            std::cout << "Decoded video frame: " << frameCount++ << std::endl;
        }

        std::cout << "Video decoding test passed!" << std::endl;
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

    FFmpeg::MediaFormat mediaFormat("C:\\Users\\tjerf\\source\\repos\\FFMPY\\Input_short.mp4");
    mediaFormat.open();

    testVideoDecoding(mediaFormat);
    testAudioDecoding(mediaFormat);

    std::cout << "Tests completed." << std::endl;
    return 0;
}
