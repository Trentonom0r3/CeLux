#include "Hqx.hpp"
#include <sstream>

Hqx::Hqx(int scaleFactor) {
    // Initialize member variables from parameters
    this->scaleFactor_ = scaleFactor;
}

Hqx::~Hqx() {
    // Destructor implementation (if needed)
}

void Hqx::setScaleFactor(int value) {
    scaleFactor_ = value;
}

int Hqx::getScaleFactor() const {
    return scaleFactor_;
}

std::string Hqx::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hqx";

    bool first = true;

    if (scaleFactor_ != 3) {
        desc << (first ? "=" : ":") << "n=" << scaleFactor_;
        first = false;
    }

    return desc.str();
}
