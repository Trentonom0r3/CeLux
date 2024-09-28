#ifndef MEDIAFORMAT_HPP
#define MEDIAFORMAT_HPP

#include "FFCore.hpp"
#include "FFException.hpp"

namespace FFmpeg {

    class MediaFormat {
    public:
        MediaFormat(const std::string& filepath);
        ~MediaFormat();

        // Open and release functions
        void open();
        const AVCodec* selectBestDecoder();
        void release();

        // Get general stream information
        int getStreamCount() const;
        std::string getFormatName() const;
        int64_t getDuration() const;

        // Stream selection
        void selectVideoStream(int streamIndex);
        void selectAudioStream(int streamIndex);

        // Video stream info
        int getVideoWidth() const;
        int getVideoHeight() const;
        std::string getVideoCodecName() const;
        double getFrameRate() const;

        // Audio stream info
        int getAudioSampleRate() const;
        int getAudioChannels() const;
        std::string getAudioCodecName() const;

        // Utility functions
        AVFormatContext* get() const; // Now public to allow access
        int getVideoStreamIndex() const; // New function to get the video stream index
        int getAudioStreamIndex() const; // New function to get the audio stream index
        std::vector<std::string> listAvailableVideoCodecs() const;
        std::vector<std::string> listAvailableAudioCodecs() const;
        std::vector<std::string> getSupportedHardwareConfigs() const;
    private:
        AVFormatContext* formatCtx_;
        AVCodecParameters* videoCodecParams_; // Store codec parameters for video stream
        AVCodecParameters* audioCodecParams_; // Store codec parameters for audio stream
        int videoStreamIndex_;
        int audioStreamIndex_;
        std::string filepath_;
    };
}

#endif // MEDIAFORMAT_HPP
