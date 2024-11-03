#include "Overlay.hpp"
#include <sstream>

Overlay::Overlay(const std::string& x, const std::string& y, int eof_action, int eval, bool shortest, int format, bool repeatlast, int alpha) {
    // Initialize member variables from parameters
    this->x_ = x;
    this->y_ = y;
    this->eof_action_ = eof_action;
    this->eval_ = eval;
    this->shortest_ = shortest;
    this->format_ = format;
    this->repeatlast_ = repeatlast;
    this->alpha_ = alpha;
}

Overlay::~Overlay() {
    // Destructor implementation (if needed)
}

void Overlay::setX(const std::string& value) {
    x_ = value;
}

std::string Overlay::getX() const {
    return x_;
}

void Overlay::setY(const std::string& value) {
    y_ = value;
}

std::string Overlay::getY() const {
    return y_;
}

void Overlay::setEof_action(int value) {
    eof_action_ = value;
}

int Overlay::getEof_action() const {
    return eof_action_;
}

void Overlay::setEval(int value) {
    eval_ = value;
}

int Overlay::getEval() const {
    return eval_;
}

void Overlay::setShortest(bool value) {
    shortest_ = value;
}

bool Overlay::getShortest() const {
    return shortest_;
}

void Overlay::setFormat(int value) {
    format_ = value;
}

int Overlay::getFormat() const {
    return format_;
}

void Overlay::setRepeatlast(bool value) {
    repeatlast_ = value;
}

bool Overlay::getRepeatlast() const {
    return repeatlast_;
}

void Overlay::setAlpha(int value) {
    alpha_ = value;
}

int Overlay::getAlpha() const {
    return alpha_;
}

std::string Overlay::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "overlay";

    bool first = true;

    if (x_ != "0") {
        desc << (first ? "=" : ":") << "x=" << x_;
        first = false;
    }
    if (y_ != "0") {
        desc << (first ? "=" : ":") << "y=" << y_;
        first = false;
    }
    if (eof_action_ != 0) {
        desc << (first ? "=" : ":") << "eof_action=" << eof_action_;
        first = false;
    }
    if (eval_ != 1) {
        desc << (first ? "=" : ":") << "eval=" << eval_;
        first = false;
    }
    if (shortest_ != false) {
        desc << (first ? "=" : ":") << "shortest=" << (shortest_ ? "1" : "0");
        first = false;
    }
    if (format_ != 0) {
        desc << (first ? "=" : ":") << "format=" << format_;
        first = false;
    }
    if (repeatlast_ != true) {
        desc << (first ? "=" : ":") << "repeatlast=" << (repeatlast_ ? "1" : "0");
        first = false;
    }
    if (alpha_ != 0) {
        desc << (first ? "=" : ":") << "alpha=" << alpha_;
        first = false;
    }

    return desc.str();
}
