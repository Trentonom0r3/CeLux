#pragma once
#ifndef DECODERS_HPP
#define DECODERS_HPP

#include <backends/Decoder.hpp>
#include <backends/cpu/Decoder.hpp>

#ifdef CUDA_ENABLED
#include <backends/gpu/cuda/Decoder.hpp>
#endif

#endif // DECODERS_HPP
