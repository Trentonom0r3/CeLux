#ifndef MEDIADECODER_HPP
#define MEDIADECODER_HPP

#include "FFCore.hpp"
#include "FFException.hpp"
#include "MediaFormat.hpp"
#include "MediaResampler.hpp"

namespace FFmpeg {

    class MediaDecoder {
    public:
        MediaDecoder(const MediaFormat& mediaFormat);
        ~MediaDecoder();

        // Initialize the decoder
        void initializeVideoDecoder(int streamIndex);
        void initializeAudioDecoder(int streamIndex);

        // Decode next video frame
        bool decodeVideoFrame(AVFrame* frame);

        // Decode next audio frame
        bool decodeAudioFrame(AVFrame* frame);
        // Initialize the audio resampler
        void initializeResampler(int outSampleRate, AVSampleFormat outSampleFormat, const AVChannelLayout& outChannelLayout);
        // Decode and resample audio frame
        bool decodeAndResampleAudioFrame(AVFrame* outputFrame);


        // Getters for codec contexts
        AVCodecContext* getVideoCodecContext() const;
        AVCodecContext* getAudioCodecContext() const;

        // Flush decoders
        void flushVideoDecoder();
        void flushAudioDecoder();

    private:
        const MediaFormat& mediaFormat_; // Reference to MediaFormat for stream access
        AVCodecContext* videoCodecCtx_;
        AVCodecContext* audioCodecCtx_;
        int videoStreamIndex_;
        int audioStreamIndex_;
        FFmpeg::MediaResampler audioResampler_; // Add resampler object
        bool resamplerInitialized_ = false;     // Track if resampler is initialized
        // Utility functions
        void release();
    };
}

#endif // MEDIADECODER_HPP
