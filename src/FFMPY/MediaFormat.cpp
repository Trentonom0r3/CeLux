#include "MediaFormat.hpp"
#include <iostream>

namespace FFmpeg {

	/**
	 * @brief Constructs a MediaFormat object.
	 * @param filepath The path to the media file.
	 */
	MediaFormat::MediaFormat(const std::string& filepath)
		: formatCtx_(nullptr), videoCodecParams_(nullptr), audioCodecParams_(nullptr),
		videoStreamIndex_(-1), audioStreamIndex_(-1), filepath_(filepath) {}

	MediaFormat::~MediaFormat() {
		release();
	}

	void MediaFormat::open() {
		FF_CHECK(avformat_open_input(&formatCtx_, filepath_.c_str(), nullptr, nullptr));
		FF_CHECK(avformat_find_stream_info(formatCtx_, nullptr));

		// Initialize stream indexes and codec parameters
		for (unsigned int i = 0; i < formatCtx_->nb_streams; i++) {
			AVStream* stream = formatCtx_->streams[i];
			AVCodecParameters* codecParams = stream->codecpar;

			if (codecParams->codec_type == AVMEDIA_TYPE_VIDEO && videoStreamIndex_ == -1) {
				videoStreamIndex_ = i;
				videoCodecParams_ = codecParams;
			}
			else if (codecParams->codec_type == AVMEDIA_TYPE_AUDIO && audioStreamIndex_ == -1) {
				audioStreamIndex_ = i;
				audioCodecParams_ = codecParams;
			}
		}

		if (videoStreamIndex_ == -1 && audioStreamIndex_ == -1) {
			throw Error::FFException("No video or audio stream found in the media file.");
		}
	}

	// Function to select the best decoder, prioritizing hardware acceleration
	const AVCodec* MediaFormat::selectBestDecoder() {
		if (!videoCodecParams_) {
			throw Error::FFException("No video stream selected.");
		}

		const AVCodec* codec = avcodec_find_decoder(videoCodecParams_->codec_id);
		if (!codec) {
			throw Error::FFException("Could not find decoder for the selected video codec.");
		}

		// Attempt to use hardware-accelerated codec if available
		const AVCodecHWConfig* hwConfig = getSuitableHWConfig(codec);
		if (hwConfig) {
			std::cout << "Using hardware-accelerated codec: "
				<< av_hwdevice_get_type_name(hwConfig->device_type) << std::endl;
			return codec;
		}
		else {
			// Fallback to software decoding if no hardware acceleration is available
			std::cout << "No hardware acceleration available, falling back to software decoding." << std::endl;
		}

		return codec;
	}

	/**
	 * @brief Gets the AVFormatContext for the opened media file.
	 * @return A pointer to the AVFormatContext.
	 */
	AVFormatContext* MediaFormat::get() const {
		return formatCtx_;
	}

	/**
	 * @brief Gets the number of streams in the media file.
	 * @return The number of streams.
	 * @throws FFException if the format context is not open.
	 */
	int MediaFormat::getStreamCount() const {
		if (!formatCtx_) throw Error::FFException("Format context is not open.");
		return formatCtx_->nb_streams;
	}

	/**
	 * @brief Gets the format name of the media file.
	 * @return The format name as a string.
	 * @throws FFException if the format context is not open.
	 */
	std::string MediaFormat::getFormatName() const {
		if (!formatCtx_) throw Error::FFException("Format context is not open.");
		return formatCtx_->iformat->name;
	}

	/**
	 * @brief Gets the duration of the media file.
	 * @return The duration in microseconds.
	 * @throws FFException if the format context is not open.
	 */
	int64_t MediaFormat::getDuration() const {
		if (!formatCtx_) throw Error::FFException("Format context is not open.");
		return formatCtx_->duration;
	}

	/**
	 * @brief Selects a video stream by its index.
	 * @param streamIndex The index of the video stream to select.
	 * @throws FFException if the stream index is invalid or the stream is not a video stream.
	 */
	void MediaFormat::selectVideoStream(int streamIndex) {
		if (streamIndex < 0 || streamIndex >= formatCtx_->nb_streams) {
			throw Error::FFException("Invalid video stream index.");
		}

		AVStream* stream = formatCtx_->streams[streamIndex];
		if (stream->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
			throw Error::FFException("Selected stream is not a video stream.");
		}

		videoStreamIndex_ = streamIndex;
		videoCodecParams_ = stream->codecpar;
	}

	/**
	 * @brief Selects an audio stream by its index.
	 * @param streamIndex The index of the audio stream to select.
	 * @throws FFException if the stream index is invalid or the stream is not an audio stream.
	 */
	void MediaFormat::selectAudioStream(int streamIndex) {
		if (streamIndex < 0 || streamIndex >= formatCtx_->nb_streams) {
			throw Error::FFException("Invalid audio stream index.");
		}

		AVStream* stream = formatCtx_->streams[streamIndex];
		if (stream->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) {
			throw Error::FFException("Selected stream is not an audio stream.");
		}

		audioStreamIndex_ = streamIndex;
		audioCodecParams_ = stream->codecpar;
	}

	/**
	 * @brief Gets the width of the selected video stream.
	 * @return The width of the video in pixels.
	 * @throws FFException if no video stream is selected.
	 */
	int MediaFormat::getVideoWidth() const {
		if (!videoCodecParams_) throw Error::FFException("No video stream selected.");
		return videoCodecParams_->width;
	}

	/**
	 * @brief Gets the height of the selected video stream.
	 * @return The height of the video in pixels.
	 * @throws FFException if no video stream is selected.
	 */
	int MediaFormat::getVideoHeight() const {
		if (!videoCodecParams_) throw Error::FFException("No video stream selected.");
		return videoCodecParams_->height;
	}

	/**
	 * @brief Gets the name of the codec used for the selected video stream.
	 * @return The name of the video codec.
	 * @throws FFException if no video stream is selected.
	 */
	std::string MediaFormat::getVideoCodecName() const {
		if (!videoCodecParams_) throw Error::FFException("No video stream selected.");
		const AVCodec* codec = avcodec_find_decoder(videoCodecParams_->codec_id);
		return codec ? codec->name : "Unknown";
	}

	/**
	 * @brief Gets the frame rate of the selected video stream.
	 * @return The frame rate as a double.
	 * @throws FFException if no video stream is selected.
	 */
	double MediaFormat::getFrameRate() const {
		if (!videoCodecParams_) throw Error::FFException("No video stream selected.");
		AVStream* stream = formatCtx_->streams[videoStreamIndex_];
		return av_q2d(stream->avg_frame_rate);
	}

	/**
	 * @brief Gets the sample rate of the selected audio stream.
	 * @return The audio sample rate in Hz.
	 * @throws FFException if no audio stream is selected.
	 */
	int MediaFormat::getAudioSampleRate() const {
		if (!audioCodecParams_) throw Error::FFException("No audio stream selected.");
		return audioCodecParams_->sample_rate;
	}

	/**
	 * @brief Gets the number of channels of the selected audio stream.
	 * @return The number of audio channels.
	 * @throws FFException if no audio stream is selected.
	 */
	int MediaFormat::getAudioChannels() const {
		if (!audioCodecParams_) throw Error::FFException("No audio stream selected.");
		return audioCodecParams_->ch_layout.nb_channels;
	}

	/**
	 * @brief Gets the name of the codec used for the selected audio stream.
	 * @return The name of the audio codec.
	 * @throws FFException if no audio stream is selected.
	 */
	std::string MediaFormat::getAudioCodecName() const {
		if (!audioCodecParams_) throw Error::FFException("No audio stream selected.");
		const AVCodec* codec = avcodec_find_decoder(audioCodecParams_->codec_id);
		return codec ? codec->name : "Unknown";
	}

	/**
	  * @brief Gets the index of the selected video stream.
	  * @return The video stream index.
	  */
	int MediaFormat::getVideoStreamIndex() const {
		return videoStreamIndex_;
	}

	/**
	 * @brief Gets the index of the selected audio stream.
	 * @return The audio stream index.
	 */
	int MediaFormat::getAudioStreamIndex() const {
		return audioStreamIndex_;
	}

	/**
	 * @brief Lists the available video codecs.
	 * @return A vector of strings containing the names of the available video codecs.
	 */
	std::vector<std::string> MediaFormat::listAvailableVideoCodecs() const {
		std::vector<std::string> codecs;
		void* iter = nullptr;
		const AVCodec* codec = nullptr;
		while ((codec = av_codec_iterate(&iter))) {
			if (av_codec_is_decoder(codec) && codec->type == AVMEDIA_TYPE_VIDEO) {
				codecs.push_back(codec->name);
			}
		}
		return codecs;
	}

	/**
	 * @brief Lists the available audio codecs.
	 * @return A vector of strings containing the names of the available audio codecs.
	 */
	std::vector<std::string> MediaFormat::listAvailableAudioCodecs() const {
		std::vector<std::string> codecs;
		void* iter = nullptr;
		const AVCodec* codec = nullptr;
		while ((codec = av_codec_iterate(&iter))) {
			if (av_codec_is_decoder(codec) && codec->type == AVMEDIA_TYPE_AUDIO) {
				codecs.push_back(codec->name);
			}
		}
		return codecs;
	}

	/**
	 * @brief Lists the available hardware configurations for the selected video codec.
	 * @return A vector of strings containing the names of the available hardware configurations.
	 * @throws FFException if no video stream is selected or the codec does not support hardware acceleration.
	 */
	std::vector<std::string> MediaFormat::getSupportedHardwareConfigs() const {
		if (!videoCodecParams_) {
			throw Error::FFException("No video stream selected.");
		}

		const AVCodec* codec = avcodec_find_decoder(videoCodecParams_->codec_id);
		if (!codec) {
			throw Error::FFException("Could not find decoder for the selected video codec.");
		}

		std::vector<std::string> hwConfigs;
		const AVCodecHWConfig* hwConfig = nullptr;
		int index = 0;

		// Iterate through all hardware configurations supported by the codec
		while ((hwConfig = avcodec_get_hw_config(codec, index++))) {
			// Add only the configurations that support hardware device context creation
			if (hwConfig->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
				const char* deviceTypeName = av_hwdevice_get_type_name(hwConfig->device_type);
				if (deviceTypeName) {
					hwConfigs.push_back(deviceTypeName);
				}
			}
		}

		return hwConfigs;
	}



	/**
	 * @brief Releases resources associated with the media file.
	 */
	void MediaFormat::release() {
		if (formatCtx_) {
			avformat_close_input(&formatCtx_);
			formatCtx_ = nullptr;
			videoCodecParams_ = nullptr;
			audioCodecParams_ = nullptr;
			videoStreamIndex_ = -1;
			audioStreamIndex_ = -1;
		}
	}
}
