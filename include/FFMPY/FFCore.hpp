// FFCore.hpp
#ifndef FFMPEG_WRAPPER_HPP
#define FFMPEG_WRAPPER_HPP

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/channel_layout.h>  // For handling channel layout information
#include <libavutil/samplefmt.h>        // For handling sample format information
#include <libswresample/swresample.h>  // Include for SwrContext and resampling functions
#include <libavutil/error.h>            // For error codes
#include <libavutil/opt.h>              // For AVOptions
#include <libavutil/imgutils.h>         // For image utilities
#include <libavutil/pixfmt.h>           // For pixel formats
	//hwaccel
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>


}


#include <string>
#include <memory>
#include <vector>
#include <iostream>

namespace FFmpeg {

	enum class VideoCodec {
		// Software Decoding Codecs
		AASC, AIC, ALIAS_PIX, AGM, AMV, ANM, ANSI, ARBC, ARGO, ASV1, ASV2, AURA, AURA2, AVRP, AVRN, AVS, AVUI,
		BETHSOFTVID, BFI, BINKVIDEO, BITPACKED, BMP, BMV_VIDEO, BRENDER_PIX, C93, CAVS, CDGRAPHICS, CDTOONS, CDXL,
		CFHD, CINEPAK, CLEARVIDEO, CLJR, CLLC, CPIA, CRI, CAMSTUDIO, CYUV, DDS, DFA, DIRAC, DNXHD, DPX, DSICINVIDEO,
		DVVIDEO, DXTORY, DXV, EACMV, EAMAD, EATGQ, EATGV, EATQI, BPS_8, ESCAPE124, ESCAPE130, FFV1, FFVHUFF, FIC,
		FITS, FLIC, FLV, FMVC, XM_4, FRAPS, FRWU, GDV, GEM, GIF, H261, H263, H263I, H263P, H264, HAP, HEVC, HNM4VIDEO,
		HQ_HQA, HQX, HUFFYUV, HYMT, IDCINVIDEO, IFF, IMM4, IMM5, INDEO2, INDEO3, INDEO4, INDEO5, INTERPLAYVIDEO,
		IPU, JPEG2000, JPEGLS, JV, KGV1, KMVC, LAGRITH, LEAD, LOCO, M101, MAGICYUV, MDEC, MEDIA100, MIMIC, MJPEG,
		MJPEGB, MMVIDEO, MOBICLIP, MOTIONPIXELS, MPEG1VIDEO, MPEG2VIDEO, MPEG4, MPEGVIDEO, MSA1, MSMPEG4V1, MSMPEG4V2,
		MSMPEG4, MSP2, MSRLE, MSS1, MSS2, MSVIDEO1, MSZH, MTS2, MV30, MVC1, MVC2, MVDV, MXPEG, NOTCHLC, NUV, PAF_VIDEO,
		PAM, PBM, PCX, PFM, PGM, PGMYUV, PGX, PHM, PHOTOC_D, PICTOR, PIXLET, PPM, PRORES, PROSUMER, PSD, PTX, QDRAW,
		QOI, QPEG, QTRLE, R10K, R210, RAWVIDEO, RL2, ROQVIDEO, RPZA, RTV1, RV10, RV20, RV30, RV40, SANM, SCPR, SGA, SGI,
		SGIRLE, SHEERVIDEO, SIMBIOSIS_IMX, SMACKVID, SMC, SMVJPEG, SNOW, SP5X, SPEEDHQ, SUNRAST, SVQ1, SVQ3, TARGA,
		TARGA_Y216, THEORA, THP, TIERTEXSEQVIDEO, TIFF, TMV, TRUEMOTION1, TRUEMOTION2, TRUEMOTION2RT, TSCC2, TXD,
		ULTIMOTION, UTVIDEO, V210, V210X, V308, V408, V410, VB, VBN, VBLE, VC1, VC1IMAGE, VCR1, VMDVIDEO, VMIX, VMNC,
		VP3, VP4, VP5, VP6, VP6A, VP6F, VP7, VP8, VP9, VQAVIDEO, VQC, VVC, WBMP, WEBP, WRAPPED_AVFRAME, WMV1, WMV2, WMV3,
		WMV3IMAGE, WNV1, XAN_WC3, XAN_WC4, XBM, XFACE, XL, XPM, XWD, Y41P, YLC, YOP, YUV4, V012, HDR, LIBVPX, LIBVPX_VP9,
		BINTEXT, XBIN, IDF, AV1,

		// Hardware Decoding Codecs
		AV1_CUVID, H264_CUVID, HEVC_CUVID, MJPEG_CUVID, MPEG1_CUVID, MPEG2_CUVID, MPEG4_CUVID, VC1_CUVID, VP8_CUVID,
		VP9_CUVID, VNULL
	};


	// Utility function to get a suitable hardware configuration for a codec
	inline const AVCodecHWConfig* getSuitableHWConfig(const AVCodec* codec) {
		const AVCodecHWConfig* hwConfig = nullptr;
		int index = 0;
		while ((hwConfig = avcodec_get_hw_config(codec, index++))) {
			if (hwConfig->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
				return hwConfig;
			}
		}
		return nullptr;  // No suitable hardware configuration found
	}

	// Utility function to log supported hardware configurations
	inline void logSupportedHardwareConfigs(const AVCodec* codec) {
		const AVCodecHWConfig* hwConfig = nullptr;
		int index = 0;
		while ((hwConfig = avcodec_get_hw_config(codec, index++))) {
			const char* deviceTypeName = av_hwdevice_get_type_name(hwConfig->device_type);
			if (deviceTypeName) {
				std::cout << "Supported hardware config: " << deviceTypeName << std::endl;
			}
		}
	}



	// Utility function to convert FFmpeg error codes to readable strings
	inline std::string errorToString(int errorCode) {
		char errBuf[AV_ERROR_MAX_STRING_SIZE];
		av_strerror(errorCode, errBuf, AV_ERROR_MAX_STRING_SIZE);
		return std::string(errBuf);
	}

	inline bool isHardwareAccelerationSupported(const AVCodec* codec) {
		// Iterate over all hardware configurations for the codec
		for (int i = 0;; i++) {
			const AVCodecHWConfig* hwConfig = avcodec_get_hw_config(codec, i);
			if (!hwConfig) {
				break; // No more configurations
			}

			// Check if the codec has any hardware acceleration capabilities
			if (hwConfig->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
				return true;
			}
		}
		return false;
	}

} // namespace FFmpeg

#endif // FFMPEG_WRAPPER_HPP
