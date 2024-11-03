#include "Greyedge.hpp"
#include <sstream>

Greyedge::Greyedge(int difford, int minknorm, double sigma) {
    // Initialize member variables from parameters
    this->difford_ = difford;
    this->minknorm_ = minknorm;
    this->sigma_ = sigma;
}

Greyedge::~Greyedge() {
    // Destructor implementation (if needed)
}

void Greyedge::setDifford(int value) {
    difford_ = value;
}

int Greyedge::getDifford() const {
    return difford_;
}

void Greyedge::setMinknorm(int value) {
    minknorm_ = value;
}

int Greyedge::getMinknorm() const {
    return minknorm_;
}

void Greyedge::setSigma(double value) {
    sigma_ = value;
}

double Greyedge::getSigma() const {
    return sigma_;
}

std::string Greyedge::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "greyedge";

    bool first = true;

    if (difford_ != 1) {
        desc << (first ? "=" : ":") << "difford=" << difford_;
        first = false;
    }
    if (minknorm_ != 1) {
        desc << (first ? "=" : ":") << "minknorm=" << minknorm_;
        first = false;
    }
    if (sigma_ != 1.00) {
        desc << (first ? "=" : ":") << "sigma=" << sigma_;
        first = false;
    }

    return desc.str();
}
