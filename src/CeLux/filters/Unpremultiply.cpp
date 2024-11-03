#include "Unpremultiply.hpp"
#include <sstream>

Unpremultiply::Unpremultiply(int planes, bool inplace) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->inplace_ = inplace;
}

Unpremultiply::~Unpremultiply() {
    // Destructor implementation (if needed)
}

void Unpremultiply::setPlanes(int value) {
    planes_ = value;
}

int Unpremultiply::getPlanes() const {
    return planes_;
}

void Unpremultiply::setInplace(bool value) {
    inplace_ = value;
}

bool Unpremultiply::getInplace() const {
    return inplace_;
}

std::string Unpremultiply::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "unpremultiply";

    bool first = true;

    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (inplace_ != false) {
        desc << (first ? "=" : ":") << "inplace=" << (inplace_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
