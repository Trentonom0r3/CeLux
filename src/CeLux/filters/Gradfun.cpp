#include "Gradfun.hpp"
#include <sstream>

Gradfun::Gradfun(float strength, int radius) {
    // Initialize member variables from parameters
    this->strength_ = strength;
    this->radius_ = radius;
}

Gradfun::~Gradfun() {
    // Destructor implementation (if needed)
}

void Gradfun::setStrength(float value) {
    strength_ = value;
}

float Gradfun::getStrength() const {
    return strength_;
}

void Gradfun::setRadius(int value) {
    radius_ = value;
}

int Gradfun::getRadius() const {
    return radius_;
}

std::string Gradfun::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "gradfun";

    bool first = true;

    if (strength_ != 1.20) {
        desc << (first ? "=" : ":") << "strength=" << strength_;
        first = false;
    }
    if (radius_ != 16) {
        desc << (first ? "=" : ":") << "radius=" << radius_;
        first = false;
    }

    return desc.str();
}
