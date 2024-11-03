#include "Entropy.hpp"
#include <sstream>

Entropy::Entropy(int mode) {
    // Initialize member variables from parameters
    this->mode_ = mode;
}

Entropy::~Entropy() {
    // Destructor implementation (if needed)
}

void Entropy::setMode(int value) {
    mode_ = value;
}

int Entropy::getMode() const {
    return mode_;
}

std::string Entropy::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "entropy";

    bool first = true;

    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }

    return desc.str();
}
