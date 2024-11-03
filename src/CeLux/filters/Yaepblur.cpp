#include "Yaepblur.hpp"
#include <sstream>

Yaepblur::Yaepblur(int radius, int planes, int sigma) {
    // Initialize member variables from parameters
    this->radius_ = radius;
    this->planes_ = planes;
    this->sigma_ = sigma;
}

Yaepblur::~Yaepblur() {
    // Destructor implementation (if needed)
}

void Yaepblur::setRadius(int value) {
    radius_ = value;
}

int Yaepblur::getRadius() const {
    return radius_;
}

void Yaepblur::setPlanes(int value) {
    planes_ = value;
}

int Yaepblur::getPlanes() const {
    return planes_;
}

void Yaepblur::setSigma(int value) {
    sigma_ = value;
}

int Yaepblur::getSigma() const {
    return sigma_;
}

std::string Yaepblur::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "yaepblur";

    bool first = true;

    if (radius_ != 3) {
        desc << (first ? "=" : ":") << "radius=" << radius_;
        first = false;
    }
    if (planes_ != 1) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (sigma_ != 128) {
        desc << (first ? "=" : ":") << "sigma=" << sigma_;
        first = false;
    }

    return desc.str();
}
