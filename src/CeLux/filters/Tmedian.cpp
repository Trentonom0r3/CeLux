#include "Tmedian.hpp"
#include <sstream>

Tmedian::Tmedian(int radius, int planes, float percentile) {
    // Initialize member variables from parameters
    this->radius_ = radius;
    this->planes_ = planes;
    this->percentile_ = percentile;
}

Tmedian::~Tmedian() {
    // Destructor implementation (if needed)
}

void Tmedian::setRadius(int value) {
    radius_ = value;
}

int Tmedian::getRadius() const {
    return radius_;
}

void Tmedian::setPlanes(int value) {
    planes_ = value;
}

int Tmedian::getPlanes() const {
    return planes_;
}

void Tmedian::setPercentile(float value) {
    percentile_ = value;
}

float Tmedian::getPercentile() const {
    return percentile_;
}

std::string Tmedian::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "tmedian";

    bool first = true;

    if (radius_ != 1) {
        desc << (first ? "=" : ":") << "radius=" << radius_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (percentile_ != 0.50) {
        desc << (first ? "=" : ":") << "percentile=" << percentile_;
        first = false;
    }

    return desc.str();
}
