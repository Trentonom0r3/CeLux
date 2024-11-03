#include "Mergeplanes.hpp"
#include <sstream>

Mergeplanes::Mergeplanes(const std::string& format, int map0s, int map0p, int map1s, int map1p, int map2s, int map2p, int map3s, int map3p) {
    // Initialize member variables from parameters
    this->format_ = format;
    this->map0s_ = map0s;
    this->map0p_ = map0p;
    this->map1s_ = map1s;
    this->map1p_ = map1p;
    this->map2s_ = map2s;
    this->map2p_ = map2p;
    this->map3s_ = map3s;
    this->map3p_ = map3p;
}

Mergeplanes::~Mergeplanes() {
    // Destructor implementation (if needed)
}

void Mergeplanes::setFormat(const std::string& value) {
    format_ = value;
}

std::string Mergeplanes::getFormat() const {
    return format_;
}

void Mergeplanes::setMap0s(int value) {
    map0s_ = value;
}

int Mergeplanes::getMap0s() const {
    return map0s_;
}

void Mergeplanes::setMap0p(int value) {
    map0p_ = value;
}

int Mergeplanes::getMap0p() const {
    return map0p_;
}

void Mergeplanes::setMap1s(int value) {
    map1s_ = value;
}

int Mergeplanes::getMap1s() const {
    return map1s_;
}

void Mergeplanes::setMap1p(int value) {
    map1p_ = value;
}

int Mergeplanes::getMap1p() const {
    return map1p_;
}

void Mergeplanes::setMap2s(int value) {
    map2s_ = value;
}

int Mergeplanes::getMap2s() const {
    return map2s_;
}

void Mergeplanes::setMap2p(int value) {
    map2p_ = value;
}

int Mergeplanes::getMap2p() const {
    return map2p_;
}

void Mergeplanes::setMap3s(int value) {
    map3s_ = value;
}

int Mergeplanes::getMap3s() const {
    return map3s_;
}

void Mergeplanes::setMap3p(int value) {
    map3p_ = value;
}

int Mergeplanes::getMap3p() const {
    return map3p_;
}

std::string Mergeplanes::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "mergeplanes";

    bool first = true;

    if (format_ != "yuva444p") {
        desc << (first ? "=" : ":") << "format=" << format_;
        first = false;
    }
    if (map0s_ != 0) {
        desc << (first ? "=" : ":") << "map0s=" << map0s_;
        first = false;
    }
    if (map0p_ != 0) {
        desc << (first ? "=" : ":") << "map0p=" << map0p_;
        first = false;
    }
    if (map1s_ != 0) {
        desc << (first ? "=" : ":") << "map1s=" << map1s_;
        first = false;
    }
    if (map1p_ != 0) {
        desc << (first ? "=" : ":") << "map1p=" << map1p_;
        first = false;
    }
    if (map2s_ != 0) {
        desc << (first ? "=" : ":") << "map2s=" << map2s_;
        first = false;
    }
    if (map2p_ != 0) {
        desc << (first ? "=" : ":") << "map2p=" << map2p_;
        first = false;
    }
    if (map3s_ != 0) {
        desc << (first ? "=" : ":") << "map3s=" << map3s_;
        first = false;
    }
    if (map3p_ != 0) {
        desc << (first ? "=" : ":") << "map3p=" << map3p_;
        first = false;
    }

    return desc.str();
}
