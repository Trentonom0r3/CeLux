#include "Xbr.hpp"
#include <sstream>

Xbr::Xbr(int scaleFactor) {
    // Initialize member variables from parameters
    this->scaleFactor_ = scaleFactor;
}

Xbr::~Xbr() {
    // Destructor implementation (if needed)
}

void Xbr::setScaleFactor(int value) {
    scaleFactor_ = value;
}

int Xbr::getScaleFactor() const {
    return scaleFactor_;
}

std::string Xbr::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "xbr";

    bool first = true;

    if (scaleFactor_ != 3) {
        desc << (first ? "=" : ":") << "n=" << scaleFactor_;
        first = false;
    }

    return desc.str();
}
