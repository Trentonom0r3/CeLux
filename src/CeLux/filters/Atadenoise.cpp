#include "Atadenoise.hpp"
#include <sstream>

Atadenoise::Atadenoise(float _0a, float _0b, float _1a, float _1b, float _2a, float _2b, int howManyFramesToUse, int whatPlanesToFilter, int variantOfAlgorithm, float _0s, float _1s, float _2s) {
    // Initialize member variables from parameters
    this->_0a_ = _0a;
    this->_0b_ = _0b;
    this->_1a_ = _1a;
    this->_1b_ = _1b;
    this->_2a_ = _2a;
    this->_2b_ = _2b;
    this->howManyFramesToUse_ = howManyFramesToUse;
    this->whatPlanesToFilter_ = whatPlanesToFilter;
    this->variantOfAlgorithm_ = variantOfAlgorithm;
    this->_0s_ = _0s;
    this->_1s_ = _1s;
    this->_2s_ = _2s;
}

Atadenoise::~Atadenoise() {
    // Destructor implementation (if needed)
}

void Atadenoise::set_0a(float value) {
    _0a_ = value;
}

float Atadenoise::get_0a() const {
    return _0a_;
}

void Atadenoise::set_0b(float value) {
    _0b_ = value;
}

float Atadenoise::get_0b() const {
    return _0b_;
}

void Atadenoise::set_1a(float value) {
    _1a_ = value;
}

float Atadenoise::get_1a() const {
    return _1a_;
}

void Atadenoise::set_1b(float value) {
    _1b_ = value;
}

float Atadenoise::get_1b() const {
    return _1b_;
}

void Atadenoise::set_2a(float value) {
    _2a_ = value;
}

float Atadenoise::get_2a() const {
    return _2a_;
}

void Atadenoise::set_2b(float value) {
    _2b_ = value;
}

float Atadenoise::get_2b() const {
    return _2b_;
}

void Atadenoise::setHowManyFramesToUse(int value) {
    howManyFramesToUse_ = value;
}

int Atadenoise::getHowManyFramesToUse() const {
    return howManyFramesToUse_;
}

void Atadenoise::setWhatPlanesToFilter(int value) {
    whatPlanesToFilter_ = value;
}

int Atadenoise::getWhatPlanesToFilter() const {
    return whatPlanesToFilter_;
}

void Atadenoise::setVariantOfAlgorithm(int value) {
    variantOfAlgorithm_ = value;
}

int Atadenoise::getVariantOfAlgorithm() const {
    return variantOfAlgorithm_;
}

void Atadenoise::set_0s(float value) {
    _0s_ = value;
}

float Atadenoise::get_0s() const {
    return _0s_;
}

void Atadenoise::set_1s(float value) {
    _1s_ = value;
}

float Atadenoise::get_1s() const {
    return _1s_;
}

void Atadenoise::set_2s(float value) {
    _2s_ = value;
}

float Atadenoise::get_2s() const {
    return _2s_;
}

std::string Atadenoise::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "atadenoise";

    bool first = true;

    if (_0a_ != 0.02) {
        desc << (first ? "=" : ":") << "0a=" << _0a_;
        first = false;
    }
    if (_0b_ != 0.04) {
        desc << (first ? "=" : ":") << "0b=" << _0b_;
        first = false;
    }
    if (_1a_ != 0.02) {
        desc << (first ? "=" : ":") << "1a=" << _1a_;
        first = false;
    }
    if (_1b_ != 0.04) {
        desc << (first ? "=" : ":") << "1b=" << _1b_;
        first = false;
    }
    if (_2a_ != 0.02) {
        desc << (first ? "=" : ":") << "2a=" << _2a_;
        first = false;
    }
    if (_2b_ != 0.04) {
        desc << (first ? "=" : ":") << "2b=" << _2b_;
        first = false;
    }
    if (howManyFramesToUse_ != 9) {
        desc << (first ? "=" : ":") << "s=" << howManyFramesToUse_;
        first = false;
    }
    if (whatPlanesToFilter_ != 7) {
        desc << (first ? "=" : ":") << "p=" << whatPlanesToFilter_;
        first = false;
    }
    if (variantOfAlgorithm_ != 0) {
        desc << (first ? "=" : ":") << "a=" << variantOfAlgorithm_;
        first = false;
    }
    if (_0s_ != 32767.00) {
        desc << (first ? "=" : ":") << "0s=" << _0s_;
        first = false;
    }
    if (_1s_ != 32767.00) {
        desc << (first ? "=" : ":") << "1s=" << _1s_;
        first = false;
    }
    if (_2s_ != 32767.00) {
        desc << (first ? "=" : ":") << "2s=" << _2s_;
        first = false;
    }

    return desc.str();
}
