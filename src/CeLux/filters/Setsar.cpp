#include "Setsar.hpp"
#include <sstream>

Setsar::Setsar(const std::string& ratio, int max) {
    // Initialize member variables from parameters
    this->ratio_ = ratio;
    this->max_ = max;
}

Setsar::~Setsar() {
    // Destructor implementation (if needed)
}

void Setsar::setRatio(const std::string& value) {
    ratio_ = value;
}

std::string Setsar::getRatio() const {
    return ratio_;
}

void Setsar::setMax(int value) {
    max_ = value;
}

int Setsar::getMax() const {
    return max_;
}

std::string Setsar::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "setsar";

    bool first = true;

    if (ratio_ != "0") {
        desc << (first ? "=" : ":") << "ratio=" << ratio_;
        first = false;
    }
    if (max_ != 100) {
        desc << (first ? "=" : ":") << "max=" << max_;
        first = false;
    }

    return desc.str();
}
