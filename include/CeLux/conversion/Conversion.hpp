#pragma once
#ifndef CONVERSION_HPP
#define CONVERSION_HPP

#ifdef CUDA_ENABLED
#include <CPUConverters.hpp>
#include <GPUConverters.hpp>
#else
#include <CPUConverters.hpp>
#endif
namespace celux
{
	enum class ConversionType
	{
		RGBToNV12,
		NV12ToRGB,
		YUV420ToRGB,
		RGBToYUV420,
	};

	enum class SupportedFormats
    {
        YUV420P,
        NV12,
        YUV420P10LE,
        P010,
		RGB24,
		RGB48
    };

	enum SupportedBitDepth
	{
		DEPTH8 = 8,
		DEPTH10 = 10,
	};

	enum class backend
    {
        CPU,
        CUDA
    };

    }


#endif

