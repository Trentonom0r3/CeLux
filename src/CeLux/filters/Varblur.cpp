#include "Varblur.hpp"
#include <sstream>

Varblur::Varblur(int min_r, int max_r, int planes) {
    // Initialize member variables from parameters
    this->min_r_ = min_r;
    this->max_r_ = max_r;
    this->planes_ = planes;
}

Varblur::~Varblur() {
    // Destructor implementation (if needed)
}

void Varblur::setMin_r(int value) {
    min_r_ = value;
}

int Varblur::getMin_r() const {
    return min_r_;
}

void Varblur::setMax_r(int value) {
    max_r_ = value;
}

int Varblur::getMax_r() const {
    return max_r_;
}

void Varblur::setPlanes(int value) {
    planes_ = value;
}

int Varblur::getPlanes() const {
    return planes_;
}

std::string Varblur::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "varblur";

    bool first = true;

    if (min_r_ != 0) {
        desc << (first ? "=" : ":") << "min_r=" << min_r_;
        first = false;
    }
    if (max_r_ != 8) {
        desc << (first ? "=" : ":") << "max_r=" << max_r_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
