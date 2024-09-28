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
    void MediaDecoder::initializeVideoDecoder(int streamIndex, const std::string& preferredCodecName) {
        const AVStream* videoStream = mediaFormat_.get()->streams[streamIndex];
        AVCodecParameters* codecParams = videoStream->codecpar;

        const AVCodec* codec = nullptr;

        // Attempt to find the codec by the preferred name if provided
        if (!preferredCodecName.empty()) {
            codec = avcodec_find_decoder_by_name(preferredCodecName.c_str());
        }

        // If the preferred codec wasn't found or no name was provided, use the default codec for the stream's codec ID
        if (!codec) {
            codec = avcodec_find_decoder(codecParams->codec_id);
            if (!codec) {
                throw Error::FFException("Unsupported video codec!");
            }
        }

        // Check for available hardware acceleration for the codec
        const AVCodecHWConfig* hwConfig = nullptr;
        int index = 0;
        while ((hwConfig = avcodec_get_hw_config(codec, index++))) {
            if (hwConfig->device_type == AV_HWDEVICE_TYPE_CUDA && (hwConfig->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) {
                // Try to use the hardware-accelerated decoder
                AVBufferRef* hw_device_ctx = nullptr;
                if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) >= 0) {
                    // Allocate the codec context
                    videoCodecCtx_ = avcodec_alloc_context3(codec);
                    if (!videoCodecCtx_) {
                        throw Error::FFException("Failed to allocate video codec context.");
                    }

                    // Set codec parameters and hardware context
                    FF_CHECK(avcodec_parameters_to_context(videoCodecCtx_, codecParams));
                    videoCodecCtx_->hw_device_ctx = av_buffer_ref(hw_device_ctx);
                    videoCodecCtx_->pix_fmt = hwConfig->pix_fmt; // Set the pixel format to the hardware-supported one
                    videoCodecCtx_->pkt_timebase = mediaFormat_.get()->streams[streamIndex]->time_base;
                    av_buffer_unref(&hw_device_ctx);

                    // Open the codec with hardware acceleration
                    if (avcodec_open2(videoCodecCtx_, codec, nullptr) == 0) {
                        std::cout << "Using hardware-accelerated decoding with pixel format: "
                            << av_get_pix_fmt_name(hwConfig->pix_fmt) << std::endl;
                        videoStreamIndex_ = streamIndex;
                        return;  // Successfully initialized with hardware acceleration
                    }
                    else {
                        std::cerr << "Failed to open video codec with hardware acceleration, falling back to software decoding." << std::endl;
                    }
                }
                else {
                    std::cerr << "Failed to create hardware device context for CUDA." << std::endl;
                }
            }
        }

        // If hardware acceleration failed, fallback to software decoding
        videoCodecCtx_ = avcodec_alloc_context3(codec);
        if (!videoCodecCtx_) {
            throw Error::FFException("Failed to allocate video codec context.");
        }

        FF_CHECK(avcodec_parameters_to_context(videoCodecCtx_, codecParams));
        videoCodecCtx_->pkt_timebase = mediaFormat_.get()->streams[streamIndex]->time_base;

        // Open the codec for software decoding
        if (avcodec_open2(videoCodecCtx_, codec, nullptr) < 0) {
            throw Error::FFException("Failed to open video codec for software decoding.");
        }

        std::cout << "Using software decoding." << std::endl;
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

        while (true) {
            int ret = av_read_frame(mediaFormat_.get(), &packet);

            // If end of stream, flush the decoder
            if (ret == AVERROR_EOF) {
                avcodec_send_packet(videoCodecCtx_, nullptr);
                ret = avcodec_receive_frame(videoCodecCtx_, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    return false; // No more frames to decode
                }
                else if (ret < 0) {
                    throw Error::FFException("Error receiving video frame from decoder.");
                }
                return true;
            }

            if (packet.stream_index == videoStreamIndex_) {
                ret = avcodec_send_packet(videoCodecCtx_, &packet);
                av_packet_unref(&packet); // Unreference the packet immediately

                if (ret < 0) {
                    throw Error::FFException("Error sending video packet to decoder.");
                }

                ret = avcodec_receive_frame(videoCodecCtx_, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    continue; // Continue until we get a frame or reach EOF
                }
                else if (ret < 0) {
                    throw Error::FFException("Error receiving video frame from decoder.");
                }

                return true; // Successfully decoded a video frame
            }

            av_packet_unref(&packet);
        }
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
