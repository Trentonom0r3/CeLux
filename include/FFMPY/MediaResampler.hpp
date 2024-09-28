#ifndef MEDIARESAMPLER_HPP
#define MEDIARESAMPLER_HPP

#include "FFCore.hpp"
#include "FFException.hpp"
#include <cstdint>

namespace FFmpeg {
    class MediaResampler {
    public:
        MediaResampler();
        ~MediaResampler();

        /**
         * @brief Initializes the resampler context.
         * @param inSampleRate Input sample rate.
         * @param outSampleRate Output sample rate.
         * @param inChannelLayout Input channel layout.
         * @param outChannelLayout Output channel layout.
         * @param inSampleFormat Input sample format.
         * @param outSampleFormat Output sample format.
         */
        void initialize(int inSampleRate, int outSampleRate,
            const AVChannelLayout& inChannelLayout, const AVChannelLayout& outChannelLayout,
            AVSampleFormat inSampleFormat, AVSampleFormat outSampleFormat);

        /**
         * @brief Resamples the input audio frame.
         * @param inputFrame The input audio frame to resample.
         * @param outputFrame The output audio frame to store the resampled data.
         * @return The number of samples in the output frame.
         */
        int resample(AVFrame* inputFrame, AVFrame* outputFrame);

        /**
         * @brief Releases the resampling context.
         */
        void release();

    private:
        SwrContext* swrCtx_; // FFmpeg resampling context
        int outSampleRate_;
        AVChannelLayout outChannelLayout_;
        AVSampleFormat outSampleFormat_;
    };
}

#endif // MEDIARESAMPLER_HPP
