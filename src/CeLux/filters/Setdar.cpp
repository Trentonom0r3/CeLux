#include "Setdar.hpp"
#include <sstream>

Setdar::Setdar(const std::string& ratio, int max) {
    // Initialize member variables from parameters
    this->ratio_ = ratio;
    this->max_ = max;
}

Setdar::~Setdar() {
    // Destructor implementation (if needed)
}

void Setdar::setRatio(const std::string& value) {
    ratio_ = value;
}

std::string Setdar::getRatio() const {
    return ratio_;
}

void Setdar::setMax(int value) {
    max_ = value;
}

int Setdar::getMax() const {
    return max_;
}

std::string Setdar::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "setdar";

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
