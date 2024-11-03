#include "Deconvolve.hpp"
#include <sstream>

Deconvolve::Deconvolve(int planes, int impulse, float noise) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->impulse_ = impulse;
    this->noise_ = noise;
}

Deconvolve::~Deconvolve() {
    // Destructor implementation (if needed)
}

void Deconvolve::setPlanes(int value) {
    planes_ = value;
}

int Deconvolve::getPlanes() const {
    return planes_;
}

void Deconvolve::setImpulse(int value) {
    impulse_ = value;
}

int Deconvolve::getImpulse() const {
    return impulse_;
}

void Deconvolve::setNoise(float value) {
    noise_ = value;
}

float Deconvolve::getNoise() const {
    return noise_;
}

std::string Deconvolve::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "deconvolve";

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
