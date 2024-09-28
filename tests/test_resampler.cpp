#include "MediaFormat.hpp"
#include "MediaDecoder.hpp"
#include "MediaResampler.hpp"
#include <iostream>

void testResamplerWithRealFile(const std::string& filePath) {
    try {
        // Initialize MediaFormat
        FFmpeg::MediaFormat mediaFormat(filePath);
        mediaFormat.open();

        // Initialize MediaDecoder
        FFmpeg::MediaDecoder decoder(mediaFormat);

        // Select and initialize the audio stream (automatically selects the first audio stream)
        int audioStreamIndex = -1;
        for (unsigned int i = 0; i < mediaFormat.getStreamCount(); ++i) {
            if (mediaFormat.get()->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioStreamIndex = i;
                break;
            }
        }

        if (audioStreamIndex == -1) {
            throw FFmpeg::Error::FFException("No audio stream found in the input file.");
        }

        decoder.initializeAudioDecoder(audioStreamIndex);

        // Initialize the output channel layout to mono
        AVChannelLayout outputChannelLayout;
        av_channel_layout_default(&outputChannelLayout, 1);  // 1 channel = mono

        // Initialize the Resampler
        decoder.initializeResampler(44100, AV_SAMPLE_FMT_S16, outputChannelLayout);

        // Create output frame for resampling
        AVFrame* outputFrame = av_frame_alloc();

        // Decode and resample audio
        while (decoder.decodeAndResampleAudioFrame(outputFrame)) {
            // Process the resampled output frame
            std::cout << "Resampled " << outputFrame->nb_samples << " samples." << std::endl;
        }

        // Free the output frame
        av_frame_free(&outputFrame);
        av_channel_layout_uninit(&outputChannelLayout);

    }
    catch (const FFmpeg::Error::FFException& e) {
        std::cerr << "Resampler test failed: " << e.what() << std::endl;
    }
}

int main() {
    std::string inputFilePath = "C:\\Users\\tjerf\\source\\repos\\FFMPY\\Input_short.mp4";
    std::cout << "Running resampler test with real file..." << std::endl;
    testResamplerWithRealFile(inputFilePath);
    std::cout << "Resampler test completed." << std::endl;
    return 0;
}
