#pragma once
#ifndef ENCODERS_HPP
#define ENCODERS_HPP

#include <backends/Encoder.hpp>
#include <backends/cpu/Encoder.hpp>

#ifdef CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#include <backends/gpu/cuda/Encoder.hpp>
#endif

#endif // ENCODERS_HPP
