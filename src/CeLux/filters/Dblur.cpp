#include "Dblur.hpp"
#include <sstream>

Dblur::Dblur(float angle, float radius, int planes) {
    // Initialize member variables from parameters
    this->angle_ = angle;
    this->radius_ = radius;
    this->planes_ = planes;
}

Dblur::~Dblur() {
    // Destructor implementation (if needed)
}

void Dblur::setAngle(float value) {
    angle_ = value;
}

float Dblur::getAngle() const {
    return angle_;
}

void Dblur::setRadius(float value) {
    radius_ = value;
}

float Dblur::getRadius() const {
    return radius_;
}

void Dblur::setPlanes(int value) {
    planes_ = value;
}

int Dblur::getPlanes() const {
    return planes_;
}

std::string Dblur::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "dblur";

    bool first = true;

    if (angle_ != 45.00) {
        desc << (first ? "=" : ":") << "angle=" << angle_;
        first = false;
    }
    if (radius_ != 5.00) {
        desc << (first ? "=" : ":") << "radius=" << radius_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
