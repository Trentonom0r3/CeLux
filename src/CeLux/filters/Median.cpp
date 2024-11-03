#include "Median.hpp"
#include <sstream>

Median::Median(int radius, int planes, int radiusV, float percentile) {
    // Initialize member variables from parameters
    this->radius_ = radius;
    this->planes_ = planes;
    this->radiusV_ = radiusV;
    this->percentile_ = percentile;
}

Median::~Median() {
    // Destructor implementation (if needed)
}

void Median::setRadius(int value) {
    radius_ = value;
}

int Median::getRadius() const {
    return radius_;
}

void Median::setPlanes(int value) {
    planes_ = value;
}

int Median::getPlanes() const {
    return planes_;
}

void Median::setRadiusV(int value) {
    radiusV_ = value;
}

int Median::getRadiusV() const {
    return radiusV_;
}

void Median::setPercentile(float value) {
    percentile_ = value;
}

float Median::getPercentile() const {
    return percentile_;
}

std::string Median::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "median";

    bool first = true;

    if (radius_ != 1) {
        desc << (first ? "=" : ":") << "radius=" << radius_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (radiusV_ != 0) {
        desc << (first ? "=" : ":") << "radiusV=" << radiusV_;
        first = false;
    }
    if (percentile_ != 0.50) {
        desc << (first ? "=" : ":") << "percentile=" << percentile_;
        first = false;
    }

    return desc.str();
}
