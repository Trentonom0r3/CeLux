#pragma once
#ifndef DECODERS_HPP
#define DECODERS_HPP

#include <backends/Decoder.hpp>
#include <backends/cpu/Decoder.hpp>

#ifdef CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#include <backends/gpu/cuda/Decoder.hpp>
#endif

#endif // DECODERS_HPP
