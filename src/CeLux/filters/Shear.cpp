#include "Shear.hpp"
#include <sstream>

Shear::Shear(float shx, float shy, const std::string& fillcolor, int interp) {
    // Initialize member variables from parameters
    this->shx_ = shx;
    this->shy_ = shy;
    this->fillcolor_ = fillcolor;
    this->interp_ = interp;
}

Shear::~Shear() {
    // Destructor implementation (if needed)
}

void Shear::setShx(float value) {
    shx_ = value;
}

float Shear::getShx() const {
    return shx_;
}

void Shear::setShy(float value) {
    shy_ = value;
}

float Shear::getShy() const {
    return shy_;
}

void Shear::setFillcolor(const std::string& value) {
    fillcolor_ = value;
}

std::string Shear::getFillcolor() const {
    return fillcolor_;
}

void Shear::setInterp(int value) {
    interp_ = value;
}

int Shear::getInterp() const {
    return interp_;
}

std::string Shear::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "shear";

    bool first = true;

    if (shx_ != 0.00) {
        desc << (first ? "=" : ":") << "shx=" << shx_;
        first = false;
    }
    if (shy_ != 0.00) {
        desc << (first ? "=" : ":") << "shy=" << shy_;
        first = false;
    }
    if (fillcolor_ != "black") {
        desc << (first ? "=" : ":") << "fillcolor=" << fillcolor_;
        first = false;
    }
    if (interp_ != 1) {
        desc << (first ? "=" : ":") << "interp=" << interp_;
        first = false;
    }

    return desc.str();
}
