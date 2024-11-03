#include "Pseudocolor.hpp"
#include <sstream>

Pseudocolor::Pseudocolor(const std::string& c0, const std::string& c1, const std::string& c2, const std::string& c3, int index, int preset, float opacity) {
    // Initialize member variables from parameters
    this->c0_ = c0;
    this->c1_ = c1;
    this->c2_ = c2;
    this->c3_ = c3;
    this->index_ = index;
    this->preset_ = preset;
    this->opacity_ = opacity;
}

Pseudocolor::~Pseudocolor() {
    // Destructor implementation (if needed)
}

void Pseudocolor::setC0(const std::string& value) {
    c0_ = value;
}

std::string Pseudocolor::getC0() const {
    return c0_;
}

void Pseudocolor::setC1(const std::string& value) {
    c1_ = value;
}

std::string Pseudocolor::getC1() const {
    return c1_;
}

void Pseudocolor::setC2(const std::string& value) {
    c2_ = value;
}

std::string Pseudocolor::getC2() const {
    return c2_;
}

void Pseudocolor::setC3(const std::string& value) {
    c3_ = value;
}

std::string Pseudocolor::getC3() const {
    return c3_;
}

void Pseudocolor::setIndex(int value) {
    index_ = value;
}

int Pseudocolor::getIndex() const {
    return index_;
}

void Pseudocolor::setPreset(int value) {
    preset_ = value;
}

int Pseudocolor::getPreset() const {
    return preset_;
}

void Pseudocolor::setOpacity(float value) {
    opacity_ = value;
}

float Pseudocolor::getOpacity() const {
    return opacity_;
}

std::string Pseudocolor::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "pseudocolor";

    bool first = true;

    if (c0_ != "val") {
        desc << (first ? "=" : ":") << "c0=" << c0_;
        first = false;
    }
    if (c1_ != "val") {
        desc << (first ? "=" : ":") << "c1=" << c1_;
        first = false;
    }
    if (c2_ != "val") {
        desc << (first ? "=" : ":") << "c2=" << c2_;
        first = false;
    }
    if (c3_ != "val") {
        desc << (first ? "=" : ":") << "c3=" << c3_;
        first = false;
    }
    if (index_ != 0) {
        desc << (first ? "=" : ":") << "index=" << index_;
        first = false;
    }
    if (preset_ != -1) {
        desc << (first ? "=" : ":") << "preset=" << preset_;
        first = false;
    }
    if (opacity_ != 1.00) {
        desc << (first ? "=" : ":") << "opacity=" << opacity_;
        first = false;
    }

    return desc.str();
}
