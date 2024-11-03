#include "Huesaturation.hpp"
#include <sstream>

Huesaturation::Huesaturation(float hue, float saturation, float intensity, int colors, float strength, float rw, float gw, float bw, bool lightness) {
    // Initialize member variables from parameters
    this->hue_ = hue;
    this->saturation_ = saturation;
    this->intensity_ = intensity;
    this->colors_ = colors;
    this->strength_ = strength;
    this->rw_ = rw;
    this->gw_ = gw;
    this->bw_ = bw;
    this->lightness_ = lightness;
}

Huesaturation::~Huesaturation() {
    // Destructor implementation (if needed)
}

void Huesaturation::setHue(float value) {
    hue_ = value;
}

float Huesaturation::getHue() const {
    return hue_;
}

void Huesaturation::setSaturation(float value) {
    saturation_ = value;
}

float Huesaturation::getSaturation() const {
    return saturation_;
}

void Huesaturation::setIntensity(float value) {
    intensity_ = value;
}

float Huesaturation::getIntensity() const {
    return intensity_;
}

void Huesaturation::setColors(int value) {
    colors_ = value;
}

int Huesaturation::getColors() const {
    return colors_;
}

void Huesaturation::setStrength(float value) {
    strength_ = value;
}

float Huesaturation::getStrength() const {
    return strength_;
}

void Huesaturation::setRw(float value) {
    rw_ = value;
}

float Huesaturation::getRw() const {
    return rw_;
}

void Huesaturation::setGw(float value) {
    gw_ = value;
}

float Huesaturation::getGw() const {
    return gw_;
}

void Huesaturation::setBw(float value) {
    bw_ = value;
}

float Huesaturation::getBw() const {
    return bw_;
}

void Huesaturation::setLightness(bool value) {
    lightness_ = value;
}

bool Huesaturation::getLightness() const {
    return lightness_;
}

std::string Huesaturation::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "huesaturation";

    bool first = true;

    if (hue_ != 0.00) {
        desc << (first ? "=" : ":") << "hue=" << hue_;
        first = false;
    }
    if (saturation_ != 0.00) {
        desc << (first ? "=" : ":") << "saturation=" << saturation_;
        first = false;
    }
    if (intensity_ != 0.00) {
        desc << (first ? "=" : ":") << "intensity=" << intensity_;
        first = false;
    }
    if (colors_ != 63) {
        desc << (first ? "=" : ":") << "colors=" << colors_;
        first = false;
    }
    if (strength_ != 1.00) {
        desc << (first ? "=" : ":") << "strength=" << strength_;
        first = false;
    }
    if (rw_ != 0.33) {
        desc << (first ? "=" : ":") << "rw=" << rw_;
        first = false;
    }
    if (gw_ != 0.33) {
        desc << (first ? "=" : ":") << "gw=" << gw_;
        first = false;
    }
    if (bw_ != 0.33) {
        desc << (first ? "=" : ":") << "bw=" << bw_;
        first = false;
    }
    if (lightness_ != false) {
        desc << (first ? "=" : ":") << "lightness=" << (lightness_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
