#include "Xmedian.hpp"
#include <sstream>

Xmedian::Xmedian(int inputs, int planes, float percentile) {
    // Initialize member variables from parameters
    this->inputs_ = inputs;
    this->planes_ = planes;
    this->percentile_ = percentile;
}

Xmedian::~Xmedian() {
    // Destructor implementation (if needed)
}

void Xmedian::setInputs(int value) {
    inputs_ = value;
}

int Xmedian::getInputs() const {
    return inputs_;
}

void Xmedian::setPlanes(int value) {
    planes_ = value;
}

int Xmedian::getPlanes() const {
    return planes_;
}

void Xmedian::setPercentile(float value) {
    percentile_ = value;
}

float Xmedian::getPercentile() const {
    return percentile_;
}

std::string Xmedian::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "xmedian";

    bool first = true;

    if (inputs_ != 3) {
        desc << (first ? "=" : ":") << "inputs=" << inputs_;
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
