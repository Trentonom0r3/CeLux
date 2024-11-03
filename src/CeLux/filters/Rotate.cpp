#include "Rotate.hpp"
#include <sstream>

Rotate::Rotate(const std::string& angle, const std::string& out_w, const std::string& out_h, const std::string& fillcolor, bool bilinear) {
    // Initialize member variables from parameters
    this->angle_ = angle;
    this->out_w_ = out_w;
    this->out_h_ = out_h;
    this->fillcolor_ = fillcolor;
    this->bilinear_ = bilinear;
}

Rotate::~Rotate() {
    // Destructor implementation (if needed)
}

void Rotate::setAngle(const std::string& value) {
    angle_ = value;
}

std::string Rotate::getAngle() const {
    return angle_;
}

void Rotate::setOut_w(const std::string& value) {
    out_w_ = value;
}

std::string Rotate::getOut_w() const {
    return out_w_;
}

void Rotate::setOut_h(const std::string& value) {
    out_h_ = value;
}

std::string Rotate::getOut_h() const {
    return out_h_;
}

void Rotate::setFillcolor(const std::string& value) {
    fillcolor_ = value;
}

std::string Rotate::getFillcolor() const {
    return fillcolor_;
}

void Rotate::setBilinear(bool value) {
    bilinear_ = value;
}

bool Rotate::getBilinear() const {
    return bilinear_;
}

std::string Rotate::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "rotate";

    bool first = true;

    if (angle_ != "0") {
        desc << (first ? "=" : ":") << "angle=" << angle_;
        first = false;
    }
    if (out_w_ != "iw") {
        desc << (first ? "=" : ":") << "out_w=" << out_w_;
        first = false;
    }
    if (out_h_ != "ih") {
        desc << (first ? "=" : ":") << "out_h=" << out_h_;
        first = false;
    }
    if (fillcolor_ != "black") {
        desc << (first ? "=" : ":") << "fillcolor=" << fillcolor_;
        first = false;
    }
    if (bilinear_ != true) {
        desc << (first ? "=" : ":") << "bilinear=" << (bilinear_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
