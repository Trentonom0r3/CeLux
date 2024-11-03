#include "Hue.hpp"
#include <sstream>

Hue::Hue(const std::string& hueAngleDegrees, const std::string& saturation, const std::string& hueAngleRadians, const std::string& brightness) {
    // Initialize member variables from parameters
    this->hueAngleDegrees_ = hueAngleDegrees;
    this->saturation_ = saturation;
    this->hueAngleRadians_ = hueAngleRadians;
    this->brightness_ = brightness;
}

Hue::~Hue() {
    // Destructor implementation (if needed)
}

void Hue::setHueAngleDegrees(const std::string& value) {
    hueAngleDegrees_ = value;
}

std::string Hue::getHueAngleDegrees() const {
    return hueAngleDegrees_;
}

void Hue::setSaturation(const std::string& value) {
    saturation_ = value;
}

std::string Hue::getSaturation() const {
    return saturation_;
}

void Hue::setHueAngleRadians(const std::string& value) {
    hueAngleRadians_ = value;
}

std::string Hue::getHueAngleRadians() const {
    return hueAngleRadians_;
}

void Hue::setBrightness(const std::string& value) {
    brightness_ = value;
}

std::string Hue::getBrightness() const {
    return brightness_;
}

std::string Hue::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hue";

    bool first = true;

    if (!hueAngleDegrees_.empty()) {
        desc << (first ? "=" : ":") << "h=" << hueAngleDegrees_;
        first = false;
    }
    if (saturation_ != "1") {
        desc << (first ? "=" : ":") << "s=" << saturation_;
        first = false;
    }
    if (!hueAngleRadians_.empty()) {
        desc << (first ? "=" : ":") << "H=" << hueAngleRadians_;
        first = false;
    }
    if (brightness_ != "0") {
        desc << (first ? "=" : ":") << "b=" << brightness_;
        first = false;
    }

    return desc.str();
}
