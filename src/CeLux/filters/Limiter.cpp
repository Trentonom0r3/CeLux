#include "Limiter.hpp"
#include <sstream>

Limiter::Limiter(int min, int max, int planes) {
    // Initialize member variables from parameters
    this->min_ = min;
    this->max_ = max;
    this->planes_ = planes;
}

Limiter::~Limiter() {
    // Destructor implementation (if needed)
}

void Limiter::setMin(int value) {
    min_ = value;
}

int Limiter::getMin() const {
    return min_;
}

void Limiter::setMax(int value) {
    max_ = value;
}

int Limiter::getMax() const {
    return max_;
}

void Limiter::setPlanes(int value) {
    planes_ = value;
}

int Limiter::getPlanes() const {
    return planes_;
}

std::string Limiter::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "limiter";

    bool first = true;

    if (min_ != 0) {
        desc << (first ? "=" : ":") << "min=" << min_;
        first = false;
    }
    if (max_ != 65535) {
        desc << (first ? "=" : ":") << "max=" << max_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
