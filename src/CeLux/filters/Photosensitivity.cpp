#include "Photosensitivity.hpp"
#include <sstream>

Photosensitivity::Photosensitivity(int frames, float threshold, int skip, bool bypass) {
    // Initialize member variables from parameters
    this->frames_ = frames;
    this->threshold_ = threshold;
    this->skip_ = skip;
    this->bypass_ = bypass;
}

Photosensitivity::~Photosensitivity() {
    // Destructor implementation (if needed)
}

void Photosensitivity::setFrames(int value) {
    frames_ = value;
}

int Photosensitivity::getFrames() const {
    return frames_;
}

void Photosensitivity::setThreshold(float value) {
    threshold_ = value;
}

float Photosensitivity::getThreshold() const {
    return threshold_;
}

void Photosensitivity::setSkip(int value) {
    skip_ = value;
}

int Photosensitivity::getSkip() const {
    return skip_;
}

void Photosensitivity::setBypass(bool value) {
    bypass_ = value;
}

bool Photosensitivity::getBypass() const {
    return bypass_;
}

std::string Photosensitivity::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "photosensitivity";

    bool first = true;

    if (frames_ != 30) {
        desc << (first ? "=" : ":") << "frames=" << frames_;
        first = false;
    }
    if (threshold_ != 1.00) {
        desc << (first ? "=" : ":") << "threshold=" << threshold_;
        first = false;
    }
    if (skip_ != 1) {
        desc << (first ? "=" : ":") << "skip=" << skip_;
        first = false;
    }
    if (bypass_ != false) {
        desc << (first ? "=" : ":") << "bypass=" << (bypass_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
