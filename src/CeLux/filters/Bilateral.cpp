#include "Bilateral.hpp"
#include <sstream>

Bilateral::Bilateral(float sigmaS, float sigmaR, int planes) {
    // Initialize member variables from parameters
    this->sigmaS_ = sigmaS;
    this->sigmaR_ = sigmaR;
    this->planes_ = planes;
}

Bilateral::~Bilateral() {
    // Destructor implementation (if needed)
}

void Bilateral::setSigmaS(float value) {
    sigmaS_ = value;
}

float Bilateral::getSigmaS() const {
    return sigmaS_;
}

void Bilateral::setSigmaR(float value) {
    sigmaR_ = value;
}

float Bilateral::getSigmaR() const {
    return sigmaR_;
}

void Bilateral::setPlanes(int value) {
    planes_ = value;
}

int Bilateral::getPlanes() const {
    return planes_;
}

std::string Bilateral::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "bilateral";

    bool first = true;

    if (sigmaS_ != 0.10) {
        desc << (first ? "=" : ":") << "sigmaS=" << sigmaS_;
        first = false;
    }
    if (sigmaR_ != 0.10) {
        desc << (first ? "=" : ":") << "sigmaR=" << sigmaR_;
        first = false;
    }
    if (planes_ != 1) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
