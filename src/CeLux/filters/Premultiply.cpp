#include "Premultiply.hpp"
#include <sstream>

Premultiply::Premultiply(int planes, bool inplace) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->inplace_ = inplace;
}

Premultiply::~Premultiply() {
    // Destructor implementation (if needed)
}

void Premultiply::setPlanes(int value) {
    planes_ = value;
}

int Premultiply::getPlanes() const {
    return planes_;
}

void Premultiply::setInplace(bool value) {
    inplace_ = value;
}

bool Premultiply::getInplace() const {
    return inplace_;
}

std::string Premultiply::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "premultiply";

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
