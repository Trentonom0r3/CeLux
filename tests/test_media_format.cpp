#include "MediaFormat.hpp"
#include <iostream>

void testMediaFormat() {
    try {
        // Test with a valid file
        FFmpeg::MediaFormat mediaFormat("C:\\Users\\tjerf\\source\\repos\\FFMPY\\Input_short.mp4");
        mediaFormat.open();

        // Video Stream Tests
        std::cout << "Video Codec: " << mediaFormat.getVideoCodecName() << std::endl;
        std::cout << "Video Width: " << mediaFormat.getVideoWidth() << std::endl;
        std::cout << "Video Height: " << mediaFormat.getVideoHeight() << std::endl;
        std::cout << "Frame Rate: " << mediaFormat.getFrameRate() << std::endl;

        // Audio Stream Tests
        std::cout << "Audio Codec: " << mediaFormat.getAudioCodecName() << std::endl;
        std::cout << "Audio Sample Rate: " << mediaFormat.getAudioSampleRate() << " Hz" << std::endl;
        std::cout << "Audio Channels: " << mediaFormat.getAudioChannels() << std::endl;

        // Invalid File Test
        try {
            FFmpeg::MediaFormat invalidFormat("invalid/path/to/nonexistent/file.mp4");
            invalidFormat.open();
            std::cerr << "Invalid file test failed: Exception was expected but not thrown." << std::endl;
        }
        catch (const FFmpeg::Error::FFException& e) {
            std::cout << "Invalid file test passed: " << e.what() << std::endl;
        }
    }
    catch (const FFmpeg::Error::FFException& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Running MediaFormat tests..." << std::endl;
    testMediaFormat();
    std::cout << "MediaFormat tests completed." << std::endl;
    return 0;
}
