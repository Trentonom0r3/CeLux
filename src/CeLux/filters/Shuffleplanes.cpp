#include "Shuffleplanes.hpp"
#include <sstream>

Shuffleplanes::Shuffleplanes(int map0, int map1, int map2, int map3) {
    // Initialize member variables from parameters
    this->map0_ = map0;
    this->map1_ = map1;
    this->map2_ = map2;
    this->map3_ = map3;
}

Shuffleplanes::~Shuffleplanes() {
    // Destructor implementation (if needed)
}

void Shuffleplanes::setMap0(int value) {
    map0_ = value;
}

int Shuffleplanes::getMap0() const {
    return map0_;
}

void Shuffleplanes::setMap1(int value) {
    map1_ = value;
}

int Shuffleplanes::getMap1() const {
    return map1_;
}

void Shuffleplanes::setMap2(int value) {
    map2_ = value;
}

int Shuffleplanes::getMap2() const {
    return map2_;
}

void Shuffleplanes::setMap3(int value) {
    map3_ = value;
}

int Shuffleplanes::getMap3() const {
    return map3_;
}

std::string Shuffleplanes::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "shuffleplanes";

    bool first = true;

    if (map0_ != 0) {
        desc << (first ? "=" : ":") << "map0=" << map0_;
        first = false;
    }
    if (map1_ != 1) {
        desc << (first ? "=" : ":") << "map1=" << map1_;
        first = false;
    }
    if (map2_ != 2) {
        desc << (first ? "=" : ":") << "map2=" << map2_;
        first = false;
    }
    if (map3_ != 3) {
        desc << (first ? "=" : ":") << "map3=" << map3_;
        first = false;
    }

    return desc.str();
}
