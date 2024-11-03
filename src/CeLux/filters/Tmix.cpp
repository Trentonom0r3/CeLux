#include "Tmix.hpp"
#include <sstream>

Tmix::Tmix(int frames, const std::string& weights, float scale, int planes) {
    // Initialize member variables from parameters
    this->frames_ = frames;
    this->weights_ = weights;
    this->scale_ = scale;
    this->planes_ = planes;
}

Tmix::~Tmix() {
    // Destructor implementation (if needed)
}

void Tmix::setFrames(int value) {
    frames_ = value;
}

int Tmix::getFrames() const {
    return frames_;
}

void Tmix::setWeights(const std::string& value) {
    weights_ = value;
}

std::string Tmix::getWeights() const {
    return weights_;
}

void Tmix::setScale(float value) {
    scale_ = value;
}

float Tmix::getScale() const {
    return scale_;
}

void Tmix::setPlanes(int value) {
    planes_ = value;
}

int Tmix::getPlanes() const {
    return planes_;
}

std::string Tmix::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "tmix";

    bool first = true;

    if (frames_ != 3) {
        desc << (first ? "=" : ":") << "frames=" << frames_;
        first = false;
    }
    if (weights_ != "1 1 1") {
        desc << (first ? "=" : ":") << "weights=" << weights_;
        first = false;
    }
    if (scale_ != 0.00) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
