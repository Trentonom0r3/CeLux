#include "Vignette.hpp"
#include <sstream>

Vignette::Vignette(const std::string& angle, const std::string& x0, const std::string& y0, int mode, int eval, bool dither, std::pair<int, int> aspect) {
    // Initialize member variables from parameters
    this->angle_ = angle;
    this->x0_ = x0;
    this->y0_ = y0;
    this->mode_ = mode;
    this->eval_ = eval;
    this->dither_ = dither;
    this->aspect_ = aspect;
}

Vignette::~Vignette() {
    // Destructor implementation (if needed)
}

void Vignette::setAngle(const std::string& value) {
    angle_ = value;
}

std::string Vignette::getAngle() const {
    return angle_;
}

void Vignette::setX0(const std::string& value) {
    x0_ = value;
}

std::string Vignette::getX0() const {
    return x0_;
}

void Vignette::setY0(const std::string& value) {
    y0_ = value;
}

std::string Vignette::getY0() const {
    return y0_;
}

void Vignette::setMode(int value) {
    mode_ = value;
}

int Vignette::getMode() const {
    return mode_;
}

void Vignette::setEval(int value) {
    eval_ = value;
}

int Vignette::getEval() const {
    return eval_;
}

void Vignette::setDither(bool value) {
    dither_ = value;
}

bool Vignette::getDither() const {
    return dither_;
}

void Vignette::setAspect(const std::pair<int, int>& value) {
    aspect_ = value;
}

std::pair<int, int> Vignette::getAspect() const {
    return aspect_;
}

std::string Vignette::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "vignette";

    bool first = true;

    if (angle_ != "PI/5") {
        desc << (first ? "=" : ":") << "angle=" << angle_;
        first = false;
    }
    if (x0_ != "w/2") {
        desc << (first ? "=" : ":") << "x0=" << x0_;
        first = false;
    }
    if (y0_ != "h/2") {
        desc << (first ? "=" : ":") << "y0=" << y0_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (eval_ != 0) {
        desc << (first ? "=" : ":") << "eval=" << eval_;
        first = false;
    }
    if (dither_ != true) {
        desc << (first ? "=" : ":") << "dither=" << (dither_ ? "1" : "0");
        first = false;
    }
    if (aspect_.first != 0 || aspect_.second != 1) {
        desc << (first ? "=" : ":") << "aspect=" << aspect_.first << "/" << aspect_.second;
        first = false;
    }

    return desc.str();
}
