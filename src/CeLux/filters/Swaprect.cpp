#include "Swaprect.hpp"
#include <sstream>

Swaprect::Swaprect(const std::string& rectWidth, const std::string& rectHeight, const std::string& x1, const std::string& y1, const std::string& x2, const std::string& y2) {
    // Initialize member variables from parameters
    this->rectWidth_ = rectWidth;
    this->rectHeight_ = rectHeight;
    this->x1_ = x1;
    this->y1_ = y1;
    this->x2_ = x2;
    this->y2_ = y2;
}

Swaprect::~Swaprect() {
    // Destructor implementation (if needed)
}

void Swaprect::setRectWidth(const std::string& value) {
    rectWidth_ = value;
}

std::string Swaprect::getRectWidth() const {
    return rectWidth_;
}

void Swaprect::setRectHeight(const std::string& value) {
    rectHeight_ = value;
}

std::string Swaprect::getRectHeight() const {
    return rectHeight_;
}

void Swaprect::setX1(const std::string& value) {
    x1_ = value;
}

std::string Swaprect::getX1() const {
    return x1_;
}

void Swaprect::setY1(const std::string& value) {
    y1_ = value;
}

std::string Swaprect::getY1() const {
    return y1_;
}

void Swaprect::setX2(const std::string& value) {
    x2_ = value;
}

std::string Swaprect::getX2() const {
    return x2_;
}

void Swaprect::setY2(const std::string& value) {
    y2_ = value;
}

std::string Swaprect::getY2() const {
    return y2_;
}

std::string Swaprect::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "swaprect";

    bool first = true;

    if (rectWidth_ != "w/2") {
        desc << (first ? "=" : ":") << "w=" << rectWidth_;
        first = false;
    }
    if (rectHeight_ != "h/2") {
        desc << (first ? "=" : ":") << "h=" << rectHeight_;
        first = false;
    }
    if (x1_ != "w/2") {
        desc << (first ? "=" : ":") << "x1=" << x1_;
        first = false;
    }
    if (y1_ != "h/2") {
        desc << (first ? "=" : ":") << "y1=" << y1_;
        first = false;
    }
    if (x2_ != "0") {
        desc << (first ? "=" : ":") << "x2=" << x2_;
        first = false;
    }
    if (y2_ != "0") {
        desc << (first ? "=" : ":") << "y2=" << y2_;
        first = false;
    }

    return desc.str();
}
