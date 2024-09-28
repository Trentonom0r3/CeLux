#include "MediaResampler.hpp"

namespace FFmpeg {

    MediaResampler::MediaResampler()
        : swrCtx_(nullptr), outSampleRate_(0), outChannelLayout_(), outSampleFormat_(AV_SAMPLE_FMT_NONE) {
        av_channel_layout_default(&outChannelLayout_, 0); // Initialize outChannelLayout to default
    }

    MediaResampler::~MediaResampler() {
        release();
    }

    void MediaResampler::initialize(int inSampleRate, int outSampleRate,
        const AVChannelLayout& inChannelLayout, const AVChannelLayout& outChannelLayout,
        AVSampleFormat inSampleFormat, AVSampleFormat outSampleFormat) {
        // Copy channel layouts
        av_channel_layout_copy(&outChannelLayout_, &outChannelLayout);

        // Allocate the resampling context
        FF_CHECK(swr_alloc_set_opts2(
            &swrCtx_,
            &outChannelLayout_,  // Output channel layout
            outSampleFormat,     // Output sample format
            outSampleRate,       // Output sample rate
            &inChannelLayout,    // Input channel layout
            inSampleFormat,      // Input sample format
            inSampleRate,        // Input sample rate
            0,                   // No logging
            nullptr              // No log context
        ));

        if (!swrCtx_) {
            throw FFmpeg::Error::FFException("Could not allocate resampler context.");
        }

        // Initialize the resampling context
        FF_CHECK(swr_init(swrCtx_));
        outSampleRate_ = outSampleRate;
        outSampleFormat_ = outSampleFormat;
    }

    int MediaResampler::resample(AVFrame* inputFrame, AVFrame* outputFrame) {
        if (!swrCtx_) {
            throw Error::FFException("Resampler context is not initialized.");
        }

        // Allocate output frame
        av_channel_layout_copy(&outputFrame->ch_layout, &outChannelLayout_);
        outputFrame->sample_rate = outSampleRate_;
        outputFrame->format = outSampleFormat_;

        // Calculate the number of output samples
        int64_t dstNbSamples = av_rescale_rnd(
            swr_get_delay(swrCtx_, inputFrame->sample_rate) + inputFrame->nb_samples,
            outSampleRate_, inputFrame->sample_rate, AV_ROUND_UP
        );

        // Set the number of samples in the output frame
        outputFrame->nb_samples = static_cast<int>(dstNbSamples);

        // Allocate memory for the output frame
        int ret = av_frame_get_buffer(outputFrame, 0);
        if (ret < 0) {
            throw Error::FFException("Error allocating buffer for output frame: " + std::to_string(ret));
        }

        // Perform the resampling
        int nbSamples = swr_convert(swrCtx_,
            outputFrame->data, outputFrame->nb_samples,
            (const uint8_t**)inputFrame->data, inputFrame->nb_samples);

        if (nbSamples < 0) {
            throw Error::FFException("Error during resampling.");
        }

        return nbSamples;
    }

    void MediaResampler::release() {
        if (swrCtx_) {
            swr_free(&swrCtx_);
            swrCtx_ = nullptr;
        }
        av_channel_layout_uninit(&outChannelLayout_);
    }
}
