#include "Gblur.hpp"
#include <sstream>

Gblur::Gblur(float sigma, int steps, int planes, float sigmaV) {
    // Initialize member variables from parameters
    this->sigma_ = sigma;
    this->steps_ = steps;
    this->planes_ = planes;
    this->sigmaV_ = sigmaV;
}

Gblur::~Gblur() {
    // Destructor implementation (if needed)
}

void Gblur::setSigma(float value) {
    sigma_ = value;
}

float Gblur::getSigma() const {
    return sigma_;
}

void Gblur::setSteps(int value) {
    steps_ = value;
}

int Gblur::getSteps() const {
    return steps_;
}

void Gblur::setPlanes(int value) {
    planes_ = value;
}

int Gblur::getPlanes() const {
    return planes_;
}

void Gblur::setSigmaV(float value) {
    sigmaV_ = value;
}

float Gblur::getSigmaV() const {
    return sigmaV_;
}

std::string Gblur::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "gblur";

    bool first = true;

    if (sigma_ != 0.50) {
        desc << (first ? "=" : ":") << "sigma=" << sigma_;
        first = false;
    }
    if (steps_ != 1) {
        desc << (first ? "=" : ":") << "steps=" << steps_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (sigmaV_ != -1.00) {
        desc << (first ? "=" : ":") << "sigmaV=" << sigmaV_;
        first = false;
    }

    return desc.str();
}
