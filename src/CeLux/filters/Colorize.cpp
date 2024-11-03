#include "Colorize.hpp"
#include <sstream>

Colorize::Colorize(float hue, float saturation, float lightness, float mix) {
    // Initialize member variables from parameters
    this->hue_ = hue;
    this->saturation_ = saturation;
    this->lightness_ = lightness;
    this->mix_ = mix;
}

Colorize::~Colorize() {
    // Destructor implementation (if needed)
}

void Colorize::setHue(float value) {
    hue_ = value;
}

float Colorize::getHue() const {
    return hue_;
}

void Colorize::setSaturation(float value) {
    saturation_ = value;
}

float Colorize::getSaturation() const {
    return saturation_;
}

void Colorize::setLightness(float value) {
    lightness_ = value;
}

float Colorize::getLightness() const {
    return lightness_;
}

void Colorize::setMix(float value) {
    mix_ = value;
}

float Colorize::getMix() const {
    return mix_;
}

std::string Colorize::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorize";

    bool first = true;

    if (hue_ != 0.00) {
        desc << (first ? "=" : ":") << "hue=" << hue_;
        first = false;
    }
    if (saturation_ != 0.50) {
        desc << (first ? "=" : ":") << "saturation=" << saturation_;
        first = false;
    }
    if (lightness_ != 0.50) {
        desc << (first ? "=" : ":") << "lightness=" << lightness_;
        first = false;
    }
    if (mix_ != 1.00) {
        desc << (first ? "=" : ":") << "mix=" << mix_;
        first = false;
    }

    return desc.str();
}
