#include "Cas.hpp"
#include <sstream>

Cas::Cas(float strength, int planes) {
    // Initialize member variables from parameters
    this->strength_ = strength;
    this->planes_ = planes;
}

Cas::~Cas() {
    // Destructor implementation (if needed)
}

void Cas::setStrength(float value) {
    strength_ = value;
}

float Cas::getStrength() const {
    return strength_;
}

void Cas::setPlanes(int value) {
    planes_ = value;
}

int Cas::getPlanes() const {
    return planes_;
}

std::string Cas::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "cas";

    bool first = true;

    if (strength_ != 0.00) {
        desc << (first ? "=" : ":") << "strength=" << strength_;
        first = false;
    }
    if (planes_ != 7) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
