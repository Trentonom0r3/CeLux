#include "Despill.hpp"
#include <sstream>

Despill::Despill(int type, float mix, float expand, float red, float green, float blue, float brightness, bool alpha) {
    // Initialize member variables from parameters
    this->type_ = type;
    this->mix_ = mix;
    this->expand_ = expand;
    this->red_ = red;
    this->green_ = green;
    this->blue_ = blue;
    this->brightness_ = brightness;
    this->alpha_ = alpha;
}

Despill::~Despill() {
    // Destructor implementation (if needed)
}

void Despill::setType(int value) {
    type_ = value;
}

int Despill::getType() const {
    return type_;
}

void Despill::setMix(float value) {
    mix_ = value;
}

float Despill::getMix() const {
    return mix_;
}

void Despill::setExpand(float value) {
    expand_ = value;
}

float Despill::getExpand() const {
    return expand_;
}

void Despill::setRed(float value) {
    red_ = value;
}

float Despill::getRed() const {
    return red_;
}

void Despill::setGreen(float value) {
    green_ = value;
}

float Despill::getGreen() const {
    return green_;
}

void Despill::setBlue(float value) {
    blue_ = value;
}

float Despill::getBlue() const {
    return blue_;
}

void Despill::setBrightness(float value) {
    brightness_ = value;
}

float Despill::getBrightness() const {
    return brightness_;
}

void Despill::setAlpha(bool value) {
    alpha_ = value;
}

bool Despill::getAlpha() const {
    return alpha_;
}

std::string Despill::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "despill";

    bool first = true;

    if (type_ != 0) {
        desc << (first ? "=" : ":") << "type=" << type_;
        first = false;
    }
    if (mix_ != 0.50) {
        desc << (first ? "=" : ":") << "mix=" << mix_;
        first = false;
    }
    if (expand_ != 0.00) {
        desc << (first ? "=" : ":") << "expand=" << expand_;
        first = false;
    }
    if (red_ != 0.00) {
        desc << (first ? "=" : ":") << "red=" << red_;
        first = false;
    }
    if (green_ != -1.00) {
        desc << (first ? "=" : ":") << "green=" << green_;
        first = false;
    }
    if (blue_ != 0.00) {
        desc << (first ? "=" : ":") << "blue=" << blue_;
        first = false;
    }
    if (brightness_ != 0.00) {
        desc << (first ? "=" : ":") << "brightness=" << brightness_;
        first = false;
    }
    if (alpha_ != false) {
        desc << (first ? "=" : ":") << "alpha=" << (alpha_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
