#include "Limitdiff.hpp"
#include <sstream>

Limitdiff::Limitdiff(float threshold, float elasticity, bool reference, int planes) {
    // Initialize member variables from parameters
    this->threshold_ = threshold;
    this->elasticity_ = elasticity;
    this->reference_ = reference;
    this->planes_ = planes;
}

Limitdiff::~Limitdiff() {
    // Destructor implementation (if needed)
}

void Limitdiff::setThreshold(float value) {
    threshold_ = value;
}

float Limitdiff::getThreshold() const {
    return threshold_;
}

void Limitdiff::setElasticity(float value) {
    elasticity_ = value;
}

float Limitdiff::getElasticity() const {
    return elasticity_;
}

void Limitdiff::setReference(bool value) {
    reference_ = value;
}

bool Limitdiff::getReference() const {
    return reference_;
}

void Limitdiff::setPlanes(int value) {
    planes_ = value;
}

int Limitdiff::getPlanes() const {
    return planes_;
}

std::string Limitdiff::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "limitdiff";

    bool first = true;

    if (threshold_ != 0.00) {
        desc << (first ? "=" : ":") << "threshold=" << threshold_;
        first = false;
    }
    if (elasticity_ != 2.00) {
        desc << (first ? "=" : ":") << "elasticity=" << elasticity_;
        first = false;
    }
    if (reference_ != false) {
        desc << (first ? "=" : ":") << "reference=" << (reference_ ? "1" : "0");
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
