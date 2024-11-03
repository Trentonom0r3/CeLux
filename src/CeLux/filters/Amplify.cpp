#include "Amplify.hpp"
#include <sstream>

Amplify::Amplify(int radius, float factor, float threshold, float tolerance, float low, float high, int planes) {
    // Initialize member variables from parameters
    this->radius_ = radius;
    this->factor_ = factor;
    this->threshold_ = threshold;
    this->tolerance_ = tolerance;
    this->low_ = low;
    this->high_ = high;
    this->planes_ = planes;
}

Amplify::~Amplify() {
    // Destructor implementation (if needed)
}

void Amplify::setRadius(int value) {
    radius_ = value;
}

int Amplify::getRadius() const {
    return radius_;
}

void Amplify::setFactor(float value) {
    factor_ = value;
}

float Amplify::getFactor() const {
    return factor_;
}

void Amplify::setThreshold(float value) {
    threshold_ = value;
}

float Amplify::getThreshold() const {
    return threshold_;
}

void Amplify::setTolerance(float value) {
    tolerance_ = value;
}

float Amplify::getTolerance() const {
    return tolerance_;
}

void Amplify::setLow(float value) {
    low_ = value;
}

float Amplify::getLow() const {
    return low_;
}

void Amplify::setHigh(float value) {
    high_ = value;
}

float Amplify::getHigh() const {
    return high_;
}

void Amplify::setPlanes(int value) {
    planes_ = value;
}

int Amplify::getPlanes() const {
    return planes_;
}

std::string Amplify::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "amplify";

    bool first = true;

    if (radius_ != 2) {
        desc << (first ? "=" : ":") << "radius=" << radius_;
        first = false;
    }
    if (factor_ != 2.00) {
        desc << (first ? "=" : ":") << "factor=" << factor_;
        first = false;
    }
    if (threshold_ != 10.00) {
        desc << (first ? "=" : ":") << "threshold=" << threshold_;
        first = false;
    }
    if (tolerance_ != 0.00) {
        desc << (first ? "=" : ":") << "tolerance=" << tolerance_;
        first = false;
    }
    if (low_ != 65535.00) {
        desc << (first ? "=" : ":") << "low=" << low_;
        first = false;
    }
    if (high_ != 65535.00) {
        desc << (first ? "=" : ":") << "high=" << high_;
        first = false;
    }
    if (planes_ != 7) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
