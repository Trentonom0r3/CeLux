#include "Colortemperature.hpp"
#include <sstream>

Colortemperature::Colortemperature(float temperature, float mix, float pl) {
    // Initialize member variables from parameters
    this->temperature_ = temperature;
    this->mix_ = mix;
    this->pl_ = pl;
}

Colortemperature::~Colortemperature() {
    // Destructor implementation (if needed)
}

void Colortemperature::setTemperature(float value) {
    temperature_ = value;
}

float Colortemperature::getTemperature() const {
    return temperature_;
}

void Colortemperature::setMix(float value) {
    mix_ = value;
}

float Colortemperature::getMix() const {
    return mix_;
}

void Colortemperature::setPl(float value) {
    pl_ = value;
}

float Colortemperature::getPl() const {
    return pl_;
}

std::string Colortemperature::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colortemperature";

    bool first = true;

    if (temperature_ != 6500.00) {
        desc << (first ? "=" : ":") << "temperature=" << temperature_;
        first = false;
    }
    if (mix_ != 1.00) {
        desc << (first ? "=" : ":") << "mix=" << mix_;
        first = false;
    }
    if (pl_ != 0.00) {
        desc << (first ? "=" : ":") << "pl=" << pl_;
        first = false;
    }

    return desc.str();
}
