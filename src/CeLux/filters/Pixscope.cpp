#include "Pixscope.hpp"
#include <sstream>

Pixscope::Pixscope(float scopeXOffset, float scopeYOffset, int scopeWidth, int scopeHeight, float windowOpacity, float wx, float wy) {
    // Initialize member variables from parameters
    this->scopeXOffset_ = scopeXOffset;
    this->scopeYOffset_ = scopeYOffset;
    this->scopeWidth_ = scopeWidth;
    this->scopeHeight_ = scopeHeight;
    this->windowOpacity_ = windowOpacity;
    this->wx_ = wx;
    this->wy_ = wy;
}

Pixscope::~Pixscope() {
    // Destructor implementation (if needed)
}

void Pixscope::setScopeXOffset(float value) {
    scopeXOffset_ = value;
}

float Pixscope::getScopeXOffset() const {
    return scopeXOffset_;
}

void Pixscope::setScopeYOffset(float value) {
    scopeYOffset_ = value;
}

float Pixscope::getScopeYOffset() const {
    return scopeYOffset_;
}

void Pixscope::setScopeWidth(int value) {
    scopeWidth_ = value;
}

int Pixscope::getScopeWidth() const {
    return scopeWidth_;
}

void Pixscope::setScopeHeight(int value) {
    scopeHeight_ = value;
}

int Pixscope::getScopeHeight() const {
    return scopeHeight_;
}

void Pixscope::setWindowOpacity(float value) {
    windowOpacity_ = value;
}

float Pixscope::getWindowOpacity() const {
    return windowOpacity_;
}

void Pixscope::setWx(float value) {
    wx_ = value;
}

float Pixscope::getWx() const {
    return wx_;
}

void Pixscope::setWy(float value) {
    wy_ = value;
}

float Pixscope::getWy() const {
    return wy_;
}

std::string Pixscope::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "pixscope";

    bool first = true;

    if (scopeXOffset_ != 0.50) {
        desc << (first ? "=" : ":") << "x=" << scopeXOffset_;
        first = false;
    }
    if (scopeYOffset_ != 0.50) {
        desc << (first ? "=" : ":") << "y=" << scopeYOffset_;
        first = false;
    }
    if (scopeWidth_ != 7) {
        desc << (first ? "=" : ":") << "w=" << scopeWidth_;
        first = false;
    }
    if (scopeHeight_ != 7) {
        desc << (first ? "=" : ":") << "h=" << scopeHeight_;
        first = false;
    }
    if (windowOpacity_ != 0.50) {
        desc << (first ? "=" : ":") << "o=" << windowOpacity_;
        first = false;
    }
    if (wx_ != -1.00) {
        desc << (first ? "=" : ":") << "wx=" << wx_;
        first = false;
    }
    if (wy_ != -1.00) {
        desc << (first ? "=" : ":") << "wy=" << wy_;
        first = false;
    }

    return desc.str();
}
