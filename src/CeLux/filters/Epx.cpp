#include "Epx.hpp"
#include <sstream>

Epx::Epx(int scaleFactor) {
    // Initialize member variables from parameters
    this->scaleFactor_ = scaleFactor;
}

Epx::~Epx() {
    // Destructor implementation (if needed)
}

void Epx::setScaleFactor(int value) {
    scaleFactor_ = value;
}

int Epx::getScaleFactor() const {
    return scaleFactor_;
}

std::string Epx::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "epx";

    bool first = true;

    if (scaleFactor_ != 3) {
        desc << (first ? "=" : ":") << "n=" << scaleFactor_;
        first = false;
    }

    return desc.str();
}
