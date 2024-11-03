#include "A3dscope.hpp"
#include <sstream>

A3dscope::A3dscope(std::pair<int, int> rate, std::pair<int, int> size, float fov, float roll, float pitch, float yaw, float zzoom, float zpos, int length) {
    // Initialize member variables from parameters
    this->rate_ = rate;
    this->size_ = size;
    this->fov_ = fov;
    this->roll_ = roll;
    this->pitch_ = pitch;
    this->yaw_ = yaw;
    this->zzoom_ = zzoom;
    this->zpos_ = zpos;
    this->length_ = length;
}

A3dscope::~A3dscope() {
    // Destructor implementation (if needed)
}

void A3dscope::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> A3dscope::getRate() const {
    return rate_;
}

void A3dscope::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> A3dscope::getSize() const {
    return size_;
}

void A3dscope::setFov(float value) {
    fov_ = value;
}

float A3dscope::getFov() const {
    return fov_;
}

void A3dscope::setRoll(float value) {
    roll_ = value;
}

float A3dscope::getRoll() const {
    return roll_;
}

void A3dscope::setPitch(float value) {
    pitch_ = value;
}

float A3dscope::getPitch() const {
    return pitch_;
}

void A3dscope::setYaw(float value) {
    yaw_ = value;
}

float A3dscope::getYaw() const {
    return yaw_;
}

void A3dscope::setZzoom(float value) {
    zzoom_ = value;
}

float A3dscope::getZzoom() const {
    return zzoom_;
}

void A3dscope::setZpos(float value) {
    zpos_ = value;
}

float A3dscope::getZpos() const {
    return zpos_;
}

void A3dscope::setLength(int value) {
    length_ = value;
}

int A3dscope::getLength() const {
    return length_;
}

std::string A3dscope::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "a3dscope";

    bool first = true;

    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (fov_ != 90.00) {
        desc << (first ? "=" : ":") << "fov=" << fov_;
        first = false;
    }
    if (roll_ != 0.00) {
        desc << (first ? "=" : ":") << "roll=" << roll_;
        first = false;
    }
    if (pitch_ != 0.00) {
        desc << (first ? "=" : ":") << "pitch=" << pitch_;
        first = false;
    }
    if (yaw_ != 0.00) {
        desc << (first ? "=" : ":") << "yaw=" << yaw_;
        first = false;
    }
    if (zzoom_ != 1.00) {
        desc << (first ? "=" : ":") << "zzoom=" << zzoom_;
        first = false;
    }
    if (zpos_ != 0.00) {
        desc << (first ? "=" : ":") << "zpos=" << zpos_;
        first = false;
    }
    if (length_ != 15) {
        desc << (first ? "=" : ":") << "length=" << length_;
        first = false;
    }

    return desc.str();
}
