#include "Noise.hpp"
#include <sstream>

Noise::Noise(int all_seed, int all_strength, int all_flags, int c0_flags, int c1_seed, int c1_strength, int c1_flags, int c2_seed, int c2_strength, int c2_flags, int c3_seed, int c3_strength, int c3_flags) {
    // Initialize member variables from parameters
    this->all_seed_ = all_seed;
    this->all_strength_ = all_strength;
    this->all_flags_ = all_flags;
    this->c0_flags_ = c0_flags;
    this->c1_seed_ = c1_seed;
    this->c1_strength_ = c1_strength;
    this->c1_flags_ = c1_flags;
    this->c2_seed_ = c2_seed;
    this->c2_strength_ = c2_strength;
    this->c2_flags_ = c2_flags;
    this->c3_seed_ = c3_seed;
    this->c3_strength_ = c3_strength;
    this->c3_flags_ = c3_flags;
}

Noise::~Noise() {
    // Destructor implementation (if needed)
}

void Noise::setAll_seed(int value) {
    all_seed_ = value;
}

int Noise::getAll_seed() const {
    return all_seed_;
}

void Noise::setAll_strength(int value) {
    all_strength_ = value;
}

int Noise::getAll_strength() const {
    return all_strength_;
}

void Noise::setAll_flags(int value) {
    all_flags_ = value;
}

int Noise::getAll_flags() const {
    return all_flags_;
}

void Noise::setC0_flags(int value) {
    c0_flags_ = value;
}

int Noise::getC0_flags() const {
    return c0_flags_;
}

void Noise::setC1_seed(int value) {
    c1_seed_ = value;
}

int Noise::getC1_seed() const {
    return c1_seed_;
}

void Noise::setC1_strength(int value) {
    c1_strength_ = value;
}

int Noise::getC1_strength() const {
    return c1_strength_;
}

void Noise::setC1_flags(int value) {
    c1_flags_ = value;
}

int Noise::getC1_flags() const {
    return c1_flags_;
}

void Noise::setC2_seed(int value) {
    c2_seed_ = value;
}

int Noise::getC2_seed() const {
    return c2_seed_;
}

void Noise::setC2_strength(int value) {
    c2_strength_ = value;
}

int Noise::getC2_strength() const {
    return c2_strength_;
}

void Noise::setC2_flags(int value) {
    c2_flags_ = value;
}

int Noise::getC2_flags() const {
    return c2_flags_;
}

void Noise::setC3_seed(int value) {
    c3_seed_ = value;
}

int Noise::getC3_seed() const {
    return c3_seed_;
}

void Noise::setC3_strength(int value) {
    c3_strength_ = value;
}

int Noise::getC3_strength() const {
    return c3_strength_;
}

void Noise::setC3_flags(int value) {
    c3_flags_ = value;
}

int Noise::getC3_flags() const {
    return c3_flags_;
}

std::string Noise::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "noise";

    bool first = true;

    if (all_seed_ != -1) {
        desc << (first ? "=" : ":") << "all_seed=" << all_seed_;
        first = false;
    }
    if (all_strength_ != 0) {
        desc << (first ? "=" : ":") << "all_strength=" << all_strength_;
        first = false;
    }
    if (all_flags_ != 0) {
        desc << (first ? "=" : ":") << "all_flags=" << all_flags_;
        first = false;
    }
    if (c0_flags_ != 0) {
        desc << (first ? "=" : ":") << "c0_flags=" << c0_flags_;
        first = false;
    }
    if (c1_seed_ != -1) {
        desc << (first ? "=" : ":") << "c1_seed=" << c1_seed_;
        first = false;
    }
    if (c1_strength_ != 0) {
        desc << (first ? "=" : ":") << "c1_strength=" << c1_strength_;
        first = false;
    }
    if (c1_flags_ != 0) {
        desc << (first ? "=" : ":") << "c1_flags=" << c1_flags_;
        first = false;
    }
    if (c2_seed_ != -1) {
        desc << (first ? "=" : ":") << "c2_seed=" << c2_seed_;
        first = false;
    }
    if (c2_strength_ != 0) {
        desc << (first ? "=" : ":") << "c2_strength=" << c2_strength_;
        first = false;
    }
    if (c2_flags_ != 0) {
        desc << (first ? "=" : ":") << "c2_flags=" << c2_flags_;
        first = false;
    }
    if (c3_seed_ != -1) {
        desc << (first ? "=" : ":") << "c3_seed=" << c3_seed_;
        first = false;
    }
    if (c3_strength_ != 0) {
        desc << (first ? "=" : ":") << "c3_strength=" << c3_strength_;
        first = false;
    }
    if (c3_flags_ != 0) {
        desc << (first ? "=" : ":") << "c3_flags=" << c3_flags_;
        first = false;
    }

    return desc.str();
}
