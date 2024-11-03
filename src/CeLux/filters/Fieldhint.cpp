#include "Fieldhint.hpp"
#include <sstream>

Fieldhint::Fieldhint(const std::string& hint, int mode) {
    // Initialize member variables from parameters
    this->hint_ = hint;
    this->mode_ = mode;
}

Fieldhint::~Fieldhint() {
    // Destructor implementation (if needed)
}

void Fieldhint::setHint(const std::string& value) {
    hint_ = value;
}

std::string Fieldhint::getHint() const {
    return hint_;
}

void Fieldhint::setMode(int value) {
    mode_ = value;
}

int Fieldhint::getMode() const {
    return mode_;
}

std::string Fieldhint::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "fieldhint";

    bool first = true;

    if (!hint_.empty()) {
        desc << (first ? "=" : ":") << "hint=" << hint_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }

    return desc.str();
}
