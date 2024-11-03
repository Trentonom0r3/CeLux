#include "Normalize.hpp"
#include <sstream>

Normalize::Normalize(const std::string& blackpt, const std::string& whitept, int smoothing, float independence, float strength) {
    // Initialize member variables from parameters
    this->blackpt_ = blackpt;
    this->whitept_ = whitept;
    this->smoothing_ = smoothing;
    this->independence_ = independence;
    this->strength_ = strength;
}

Normalize::~Normalize() {
    // Destructor implementation (if needed)
}

void Normalize::setBlackpt(const std::string& value) {
    blackpt_ = value;
}

std::string Normalize::getBlackpt() const {
    return blackpt_;
}

void Normalize::setWhitept(const std::string& value) {
    whitept_ = value;
}

std::string Normalize::getWhitept() const {
    return whitept_;
}

void Normalize::setSmoothing(int value) {
    smoothing_ = value;
}

int Normalize::getSmoothing() const {
    return smoothing_;
}

void Normalize::setIndependence(float value) {
    independence_ = value;
}

float Normalize::getIndependence() const {
    return independence_;
}

void Normalize::setStrength(float value) {
    strength_ = value;
}

float Normalize::getStrength() const {
    return strength_;
}

std::string Normalize::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "normalize";

    bool first = true;

    if (blackpt_ != "black") {
        desc << (first ? "=" : ":") << "blackpt=" << blackpt_;
        first = false;
    }
    if (whitept_ != "white") {
        desc << (first ? "=" : ":") << "whitept=" << whitept_;
        first = false;
    }
    if (smoothing_ != 0) {
        desc << (first ? "=" : ":") << "smoothing=" << smoothing_;
        first = false;
    }
    if (independence_ != 1.00) {
        desc << (first ? "=" : ":") << "independence=" << independence_;
        first = false;
    }
    if (strength_ != 1.00) {
        desc << (first ? "=" : ":") << "strength=" << strength_;
        first = false;
    }

    return desc.str();
}
