#include "Tmidequalizer.hpp"
#include <sstream>

Tmidequalizer::Tmidequalizer(int radius, float sigma, int planes) {
    // Initialize member variables from parameters
    this->radius_ = radius;
    this->sigma_ = sigma;
    this->planes_ = planes;
}

Tmidequalizer::~Tmidequalizer() {
    // Destructor implementation (if needed)
}

void Tmidequalizer::setRadius(int value) {
    radius_ = value;
}

int Tmidequalizer::getRadius() const {
    return radius_;
}

void Tmidequalizer::setSigma(float value) {
    sigma_ = value;
}

float Tmidequalizer::getSigma() const {
    return sigma_;
}

void Tmidequalizer::setPlanes(int value) {
    planes_ = value;
}

int Tmidequalizer::getPlanes() const {
    return planes_;
}

std::string Tmidequalizer::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "tmidequalizer";

    bool first = true;

    if (radius_ != 5) {
        desc << (first ? "=" : ":") << "radius=" << radius_;
        first = false;
    }
    if (sigma_ != 0.50) {
        desc << (first ? "=" : ":") << "sigma=" << sigma_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
