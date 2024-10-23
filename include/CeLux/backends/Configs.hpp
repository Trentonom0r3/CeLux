#pragma once
#ifndef CONFIGS_HPP
#define CONFIGS_HPP

#include <CxCore.hpp>


struct CustomTensorOptions
{
    int height;
    int width;
    int channels;
    torch::Device device;
    torch::Dtype dtype;
    bool requires_grad;
    bool pinned_memory;
    bool is_contiguous;

    struct ProcessingOptions
    {
        float mul = 1.0;
        float add = 0.0;
        float div = 1.0;
        float sub = 0.0;
        float min = 0.0;
        float max = 1.0;
        bool normalize = false;
        bool clip = false;
	} processing;


    };
};

struct CustomOptions {
    int height;
    int width;
    int channels;
    torch::Device device;
    torch::Dtype dtype;
    bool requires_grad;
    bool pinned_memory;
    bool is_contiguous;

    struct advanced
    {
    std::optional<float> mul;
    std::optional<float> add;
    std::optional<float> div;
    std::optional<float> sub;
    std::optional<float> min;
    std::optional<float> max;
    bool normalize;
    bool clip;
	} advanced;

    };













#endif // CONFIGS_HPP
