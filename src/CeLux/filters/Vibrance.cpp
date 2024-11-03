#include "Vibrance.hpp"
#include <sstream>

Vibrance::Vibrance(float intensity, float rbal, float gbal, float bbal, float rlum, float glum, float blum, bool alternate) {
    // Initialize member variables from parameters
    this->intensity_ = intensity;
    this->rbal_ = rbal;
    this->gbal_ = gbal;
    this->bbal_ = bbal;
    this->rlum_ = rlum;
    this->glum_ = glum;
    this->blum_ = blum;
    this->alternate_ = alternate;
}

Vibrance::~Vibrance() {
    // Destructor implementation (if needed)
}

void Vibrance::setIntensity(float value) {
    intensity_ = value;
}

float Vibrance::getIntensity() const {
    return intensity_;
}

void Vibrance::setRbal(float value) {
    rbal_ = value;
}

float Vibrance::getRbal() const {
    return rbal_;
}

void Vibrance::setGbal(float value) {
    gbal_ = value;
}

float Vibrance::getGbal() const {
    return gbal_;
}

void Vibrance::setBbal(float value) {
    bbal_ = value;
}

float Vibrance::getBbal() const {
    return bbal_;
}

void Vibrance::setRlum(float value) {
    rlum_ = value;
}

float Vibrance::getRlum() const {
    return rlum_;
}

void Vibrance::setGlum(float value) {
    glum_ = value;
}

float Vibrance::getGlum() const {
    return glum_;
}

void Vibrance::setBlum(float value) {
    blum_ = value;
}

float Vibrance::getBlum() const {
    return blum_;
}

void Vibrance::setAlternate(bool value) {
    alternate_ = value;
}

bool Vibrance::getAlternate() const {
    return alternate_;
}

std::string Vibrance::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "vibrance";

    bool first = true;

    if (intensity_ != 0.00) {
        desc << (first ? "=" : ":") << "intensity=" << intensity_;
        first = false;
    }
    if (rbal_ != 1.00) {
        desc << (first ? "=" : ":") << "rbal=" << rbal_;
        first = false;
    }
    if (gbal_ != 1.00) {
        desc << (first ? "=" : ":") << "gbal=" << gbal_;
        first = false;
    }
    if (bbal_ != 1.00) {
        desc << (first ? "=" : ":") << "bbal=" << bbal_;
        first = false;
    }
    if (rlum_ != 0.07) {
        desc << (first ? "=" : ":") << "rlum=" << rlum_;
        first = false;
    }
    if (glum_ != 0.72) {
        desc << (first ? "=" : ":") << "glum=" << glum_;
        first = false;
    }
    if (blum_ != 0.21) {
        desc << (first ? "=" : ":") << "blum=" << blum_;
        first = false;
    }
    if (alternate_ != false) {
        desc << (first ? "=" : ":") << "alternate=" << (alternate_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
