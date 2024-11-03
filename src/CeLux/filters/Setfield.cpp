#include "Setfield.hpp"
#include <sstream>

Setfield::Setfield(int mode) {
    // Initialize member variables from parameters
    this->mode_ = mode;
}

Setfield::~Setfield() {
    // Destructor implementation (if needed)
}

void Setfield::setMode(int value) {
    mode_ = value;
}

int Setfield::getMode() const {
    return mode_;
}

std::string Setfield::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "setfield";

    bool first = true;

    if (mode_ != -1) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }

    return desc.str();
}
