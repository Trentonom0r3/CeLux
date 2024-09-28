#include "MediaDecoder.hpp"
#include <iostream>

namespace FFmpeg {

    /**
     * @brief Constructs a MediaDecoder object using the provided MediaFormat reference.
     *
     * @param mediaFormat Reference to a MediaFormat object that contains media information.
     */
    MediaDecoder::MediaDecoder(const MediaFormat& mediaFormat)
        : mediaFormat_(mediaFormat), videoCodecCtx_(nullptr), audioCodecCtx_(nullptr),
        videoStreamIndex_(-1), audioStreamIndex_(-1) {}

    /**
     * @brief Destructor for MediaDecoder. Releases allocated codec contexts.
     */
    MediaDecoder::~MediaDecoder() {
        release();
    }

    /**
     * @brief Initializes the video decoder for the specified stream index.
     *
     * @param streamIndex The index of the video stream to initialize the decoder for.
     * @throws FFException If the codec is unsupported, context allocation fails, or opening the codec fails.
     */
    void MediaDecoder::initializeVideoDecoder(int streamIndex) {
        const AVStream* videoStream = mediaFormat_.get()->streams[streamIndex];
        AVCodecParameters* codecParams = videoStream->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);
        if (!codec) {
            throw Error::FFException("Unsupported video codec!");
        }

        videoCodecCtx_ = avcodec_alloc_context3(codec);
        if (!videoCodecCtx_) {
            throw Error::FFException("Failed to allocate video codec context.");
        }

        FF_CHECK(avcodec_parameters_to_context(videoCodecCtx_, codecParams));
        FF_CHECK(avcodec_open2(videoCodecCtx_, codec, nullptr));

        videoStreamIndex_ = streamIndex;
    }

    /**
     * @brief Initializes the audio decoder for the specified stream index.
     *
     * @param streamIndex The index of the audio stream to initialize the decoder for.
     * @throws FFException If the codec is unsupported, context allocation fails, or opening the codec fails.
     */
    void MediaDecoder::initializeAudioDecoder(int streamIndex) {
        const AVStream* audioStream = mediaFormat_.get()->streams[streamIndex];
        AVCodecParameters* codecParams = audioStream->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);
        if (!codec) {
            throw Error::FFException("Unsupported audio codec!");
        }

        audioCodecCtx_ = avcodec_alloc_context3(codec);
        if (!audioCodecCtx_) {
            throw Error::FFException("Failed to allocate audio codec context.");
        }

        FF_CHECK(avcodec_parameters_to_context(audioCodecCtx_, codecParams));
        FF_CHECK(avcodec_open2(audioCodecCtx_, codec, nullptr));

        audioStreamIndex_ = streamIndex;
    }


    /**
     * @brief Decodes the next video frame.
     *
     * @param frame Pointer to an AVFrame where the decoded video frame will be stored.
     * @return true If a video frame is successfully decoded.
     * @return false If there are no more frames to decode.
     * @throws FFException If an error occurs while sending a packet or receiving a frame.
     */
    bool MediaDecoder::decodeVideoFrame(AVFrame* frame) {
        AVPacket packet;
        av_init_packet(&packet);

        while (av_read_frame(mediaFormat_.get(), &packet) >= 0) {
            if (packet.stream_index == videoStreamIndex_) {
                int ret = avcodec_send_packet(videoCodecCtx_, &packet);
                if (ret < 0) {
                    av_packet_unref(&packet);
                    throw Error::FFException("Error sending video packet to decoder.");
                }

                ret = avcodec_receive_frame(videoCodecCtx_, frame);
                av_packet_unref(&packet);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    continue; // Keep trying until we get a frame or reach EOF
                }
                else if (ret < 0) {
                    throw Error::FFException("Error receiving video frame from decoder.");
                }

                return true; // Successfully decoded a video frame
            }

            av_packet_unref(&packet);
        }

        return false; // No more frames to decode
    }

    /**
     * @brief Decodes the next audio frame.
     *
     * @param frame Pointer to an AVFrame where the decoded audio frame will be stored.
     * @return true If an audio frame is successfully decoded.
     * @return false If there are no more frames to decode.
     * @throws FFException If an error occurs while sending a packet or receiving a frame.
     */
    bool MediaDecoder::decodeAudioFrame(AVFrame* frame) {
        AVPacket packet;
        av_init_packet(&packet);

        while (av_read_frame(mediaFormat_.get(), &packet) >= 0) {
            if (packet.stream_index == audioStreamIndex_) {
                int ret = avcodec_send_packet(audioCodecCtx_, &packet);
                if (ret < 0) {
                    av_packet_unref(&packet);
                    throw Error::FFException("Error sending audio packet to decoder.");
                }

                ret = avcodec_receive_frame(audioCodecCtx_, frame);
                av_packet_unref(&packet);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    continue; // Keep trying until we get a frame or reach EOF
                }
                else if (ret < 0) {
                    throw Error::FFException("Error receiving audio frame from decoder.");
                }

                return true; // Successfully decoded an audio frame
            }

            av_packet_unref(&packet);
        }

        return false; // No more frames to decode
    }

    // MediaDecoder.cpp (additions)
    void MediaDecoder::initializeResampler(int outSampleRate, AVSampleFormat outSampleFormat, const AVChannelLayout& outChannelLayout) {
        if (!audioCodecCtx_) {
            throw Error::FFException("Audio codec context is not initialized.");
        }

        /*enum AVSampleFormat {
            AV_SAMPLE_FMT_NONE = -1,
            AV_SAMPLE_FMT_U8,          ///< unsigned 8 bits
            AV_SAMPLE_FMT_S16,         ///< signed 16 bits
            AV_SAMPLE_FMT_S32,         ///< signed 32 bits
            AV_SAMPLE_FMT_FLT,         ///< float
            AV_SAMPLE_FMT_DBL,         ///< double

            AV_SAMPLE_FMT_U8P,         ///< unsigned 8 bits, planar
            AV_SAMPLE_FMT_S16P,        ///< signed 16 bits, planar
            AV_SAMPLE_FMT_S32P,        ///< signed 32 bits, planar
            AV_SAMPLE_FMT_FLTP,        ///< float, planar
            AV_SAMPLE_FMT_DBLP,        ///< double, planar
            AV_SAMPLE_FMT_S64,         ///< signed 64 bits
            AV_SAMPLE_FMT_S64P,        ///< signed 64 bits, planar

            AV_SAMPLE_FMT_NB           ///< Number of sample formats. DO NOT USE if linking dynamically
        };
        */

        // Initialize the resampler with input and output parameters
        audioResampler_.initialize(
            audioCodecCtx_->sample_rate,
            outSampleRate,
            audioCodecCtx_->ch_layout,
            outChannelLayout,
            audioCodecCtx_->sample_fmt,
            outSampleFormat
        );
        resamplerInitialized_ = true;
    }



    bool MediaDecoder::decodeAndResampleAudioFrame(AVFrame* outputFrame) {
        if (!resamplerInitialized_) {
            throw Error::FFException("Resampler is not initialized.");
        }

        AVFrame* inputFrame = av_frame_alloc();
        bool success = decodeAudioFrame(inputFrame);
        if (!success) {
            av_frame_free(&inputFrame);
            return false; // No more frames to decode
        }

        // Resample the decoded frame
        audioResampler_.resample(inputFrame, outputFrame);
        av_frame_free(&inputFrame);
        return true;
    }


    /**
     * @brief Gets the video codec context.
     *
     * @return AVCodecContext* Pointer to the video codec context.
     */
    AVCodecContext* MediaDecoder::getVideoCodecContext() const {
        return videoCodecCtx_;
    }

    /**
     * @brief Gets the audio codec context.
     *
     * @return AVCodecContext* Pointer to the audio codec context.
     */
    AVCodecContext* MediaDecoder::getAudioCodecContext() const {
        return audioCodecCtx_;
    }

    /**
     * @brief Flushes the video decoder buffers.
     */
    void MediaDecoder::flushVideoDecoder() {
        if (videoCodecCtx_) {
            avcodec_flush_buffers(videoCodecCtx_);
        }
    }

    /**
     * @brief Flushes the audio decoder buffers.
     */
    void MediaDecoder::flushAudioDecoder() {
        if (audioCodecCtx_) {
            avcodec_flush_buffers(audioCodecCtx_);
        }
    }

    /**
     * @brief Releases the resources allocated for the video and audio codec contexts.
     */
    void MediaDecoder::release() {
        if (videoCodecCtx_) {
            avcodec_free_context(&videoCodecCtx_);
        }

        if (audioCodecCtx_) {
            avcodec_free_context(&audioCodecCtx_);
        }

        videoStreamIndex_ = -1;
        audioStreamIndex_ = -1;
    }
}
