#pragma once
#ifndef DEOCDERS_HPP
#define DEOCDERS_HPP

#include <backends/Decoder.hpp>
#include <backends/cpu/Decoder.hpp>

#ifdef CUDA_ENABLED
#include <backends/gpu/cuda/Decoder.hpp>
#endif

#endif // DEOCDERS_HPP
