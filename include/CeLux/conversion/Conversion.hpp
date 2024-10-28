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


} // namespace celux

#endif
