#include "Guided.hpp"
#include <sstream>

Guided::Guided(int radius, float eps, int mode, int sub, int guidance, int planes) {
    // Initialize member variables from parameters
    this->radius_ = radius;
    this->eps_ = eps;
    this->mode_ = mode;
    this->sub_ = sub;
    this->guidance_ = guidance;
    this->planes_ = planes;
}

Guided::~Guided() {
    // Destructor implementation (if needed)
}

void Guided::setRadius(int value) {
    radius_ = value;
}

int Guided::getRadius() const {
    return radius_;
}

void Guided::setEps(float value) {
    eps_ = value;
}

float Guided::getEps() const {
    return eps_;
}

void Guided::setMode(int value) {
    mode_ = value;
}

int Guided::getMode() const {
    return mode_;
}

void Guided::setSub(int value) {
    sub_ = value;
}

int Guided::getSub() const {
    return sub_;
}

void Guided::setGuidance(int value) {
    guidance_ = value;
}

int Guided::getGuidance() const {
    return guidance_;
}

void Guided::setPlanes(int value) {
    planes_ = value;
}

int Guided::getPlanes() const {
    return planes_;
}

std::string Guided::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "guided";

    bool first = true;

    if (radius_ != 3) {
        desc << (first ? "=" : ":") << "radius=" << radius_;
        first = false;
    }
    if (eps_ != 0.01) {
        desc << (first ? "=" : ":") << "eps=" << eps_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (sub_ != 4) {
        desc << (first ? "=" : ":") << "sub=" << sub_;
        first = false;
    }
    if (guidance_ != 0) {
        desc << (first ? "=" : ":") << "guidance=" << guidance_;
        first = false;
    }
    if (planes_ != 1) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
