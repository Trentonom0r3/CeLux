#include "Convolve.hpp"
#include <sstream>

Convolve::Convolve(int planes, int impulse, float noise) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->impulse_ = impulse;
    this->noise_ = noise;
}

Convolve::~Convolve() {
    // Destructor implementation (if needed)
}

void Convolve::setPlanes(int value) {
    planes_ = value;
}

int Convolve::getPlanes() const {
    return planes_;
}

void Convolve::setImpulse(int value) {
    impulse_ = value;
}

int Convolve::getImpulse() const {
    return impulse_;
}

void Convolve::setNoise(float value) {
    noise_ = value;
}

float Convolve::getNoise() const {
    return noise_;
}

std::string Convolve::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "convolve";

    bool first = true;

    if (planes_ != 7) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (impulse_ != 1) {
        desc << (first ? "=" : ":") << "impulse=" << impulse_;
        first = false;
    }
    if (noise_ != 0.00) {
        desc << (first ? "=" : ":") << "noise=" << noise_;
        first = false;
    }

    return desc.str();
}
