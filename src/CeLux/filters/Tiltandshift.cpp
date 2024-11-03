#include "Tiltandshift.hpp"
#include <sstream>

Tiltandshift::Tiltandshift(int tilt, int start, int end, int hold, int pad) {
    // Initialize member variables from parameters
    this->tilt_ = tilt;
    this->start_ = start;
    this->end_ = end;
    this->hold_ = hold;
    this->pad_ = pad;
}

Tiltandshift::~Tiltandshift() {
    // Destructor implementation (if needed)
}

void Tiltandshift::setTilt(int value) {
    tilt_ = value;
}

int Tiltandshift::getTilt() const {
    return tilt_;
}

void Tiltandshift::setStart(int value) {
    start_ = value;
}

int Tiltandshift::getStart() const {
    return start_;
}

void Tiltandshift::setEnd(int value) {
    end_ = value;
}

int Tiltandshift::getEnd() const {
    return end_;
}

void Tiltandshift::setHold(int value) {
    hold_ = value;
}

int Tiltandshift::getHold() const {
    return hold_;
}

void Tiltandshift::setPad(int value) {
    pad_ = value;
}

int Tiltandshift::getPad() const {
    return pad_;
}

std::string Tiltandshift::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "tiltandshift";

    bool first = true;

    if (tilt_ != 1) {
        desc << (first ? "=" : ":") << "tilt=" << tilt_;
        first = false;
    }
    if (start_ != 0) {
        desc << (first ? "=" : ":") << "start=" << start_;
        first = false;
    }
    if (end_ != 0) {
        desc << (first ? "=" : ":") << "end=" << end_;
        first = false;
    }
    if (hold_ != 0) {
        desc << (first ? "=" : ":") << "hold=" << hold_;
        first = false;
    }
    if (pad_ != 0) {
        desc << (first ? "=" : ":") << "pad=" << pad_;
        first = false;
    }

    return desc.str();
}
