#include "Maskfun.hpp"
#include <sstream>

Maskfun::Maskfun(int low, int high, int planes, int fill, int sum) {
    // Initialize member variables from parameters
    this->low_ = low;
    this->high_ = high;
    this->planes_ = planes;
    this->fill_ = fill;
    this->sum_ = sum;
}

Maskfun::~Maskfun() {
    // Destructor implementation (if needed)
}

void Maskfun::setLow(int value) {
    low_ = value;
}

int Maskfun::getLow() const {
    return low_;
}

void Maskfun::setHigh(int value) {
    high_ = value;
}

int Maskfun::getHigh() const {
    return high_;
}

void Maskfun::setPlanes(int value) {
    planes_ = value;
}

int Maskfun::getPlanes() const {
    return planes_;
}

void Maskfun::setFill(int value) {
    fill_ = value;
}

int Maskfun::getFill() const {
    return fill_;
}

void Maskfun::setSum(int value) {
    sum_ = value;
}

int Maskfun::getSum() const {
    return sum_;
}

std::string Maskfun::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "maskfun";

    bool first = true;

    if (low_ != 10) {
        desc << (first ? "=" : ":") << "low=" << low_;
        first = false;
    }
    if (high_ != 10) {
        desc << (first ? "=" : ":") << "high=" << high_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (fill_ != 0) {
        desc << (first ? "=" : ":") << "fill=" << fill_;
        first = false;
    }
    if (sum_ != 10) {
        desc << (first ? "=" : ":") << "sum=" << sum_;
        first = false;
    }

    return desc.str();
}
