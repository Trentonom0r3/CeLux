#include "Lagfun.hpp"
#include <sstream>

Lagfun::Lagfun(float decay, int planes) {
    // Initialize member variables from parameters
    this->decay_ = decay;
    this->planes_ = planes;
}

Lagfun::~Lagfun() {
    // Destructor implementation (if needed)
}

void Lagfun::setDecay(float value) {
    decay_ = value;
}

float Lagfun::getDecay() const {
    return decay_;
}

void Lagfun::setPlanes(int value) {
    planes_ = value;
}

int Lagfun::getPlanes() const {
    return planes_;
}

std::string Lagfun::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "lagfun";

    bool first = true;

    if (decay_ != 0.95) {
        desc << (first ? "=" : ":") << "decay=" << decay_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
