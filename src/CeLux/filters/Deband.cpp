#include "Deband.hpp"
#include <sstream>

Deband::Deband(float _1thr, float _2thr, float _3thr, float _4thr, int range, float direction, bool blur, bool coupling) {
    // Initialize member variables from parameters
    this->_1thr_ = _1thr;
    this->_2thr_ = _2thr;
    this->_3thr_ = _3thr;
    this->_4thr_ = _4thr;
    this->range_ = range;
    this->direction_ = direction;
    this->blur_ = blur;
    this->coupling_ = coupling;
}

Deband::~Deband() {
    // Destructor implementation (if needed)
}

void Deband::set_1thr(float value) {
    _1thr_ = value;
}

float Deband::get_1thr() const {
    return _1thr_;
}

void Deband::set_2thr(float value) {
    _2thr_ = value;
}

float Deband::get_2thr() const {
    return _2thr_;
}

void Deband::set_3thr(float value) {
    _3thr_ = value;
}

float Deband::get_3thr() const {
    return _3thr_;
}

void Deband::set_4thr(float value) {
    _4thr_ = value;
}

float Deband::get_4thr() const {
    return _4thr_;
}

void Deband::setRange(int value) {
    range_ = value;
}

int Deband::getRange() const {
    return range_;
}

void Deband::setDirection(float value) {
    direction_ = value;
}

float Deband::getDirection() const {
    return direction_;
}

void Deband::setBlur(bool value) {
    blur_ = value;
}

bool Deband::getBlur() const {
    return blur_;
}

void Deband::setCoupling(bool value) {
    coupling_ = value;
}

bool Deband::getCoupling() const {
    return coupling_;
}

std::string Deband::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "deband";

    bool first = true;

    if (_1thr_ != 0.02) {
        desc << (first ? "=" : ":") << "1thr=" << _1thr_;
        first = false;
    }
    if (_2thr_ != 0.02) {
        desc << (first ? "=" : ":") << "2thr=" << _2thr_;
        first = false;
    }
    if (_3thr_ != 0.02) {
        desc << (first ? "=" : ":") << "3thr=" << _3thr_;
        first = false;
    }
    if (_4thr_ != 0.02) {
        desc << (first ? "=" : ":") << "4thr=" << _4thr_;
        first = false;
    }
    if (range_ != 16) {
        desc << (first ? "=" : ":") << "range=" << range_;
        first = false;
    }
    if (direction_ != 6.28) {
        desc << (first ? "=" : ":") << "direction=" << direction_;
        first = false;
    }
    if (blur_ != true) {
        desc << (first ? "=" : ":") << "blur=" << (blur_ ? "1" : "0");
        first = false;
    }
    if (coupling_ != false) {
        desc << (first ? "=" : ":") << "coupling=" << (coupling_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
