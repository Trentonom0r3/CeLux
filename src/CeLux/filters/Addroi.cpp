#include "Addroi.hpp"
#include <sstream>

Addroi::Addroi(const std::string& regionDistanceFromLeftEdgeOfFrame, const std::string& regionDistanceFromTopEdgeOfFrame, const std::string& regionWidth, const std::string& regionHeight, std::pair<int, int> qoffset, bool clear) {
    // Initialize member variables from parameters
    this->regionDistanceFromLeftEdgeOfFrame_ = regionDistanceFromLeftEdgeOfFrame;
    this->regionDistanceFromTopEdgeOfFrame_ = regionDistanceFromTopEdgeOfFrame;
    this->regionWidth_ = regionWidth;
    this->regionHeight_ = regionHeight;
    this->qoffset_ = qoffset;
    this->clear_ = clear;
}

Addroi::~Addroi() {
    // Destructor implementation (if needed)
}

void Addroi::setRegionDistanceFromLeftEdgeOfFrame(const std::string& value) {
    regionDistanceFromLeftEdgeOfFrame_ = value;
}

std::string Addroi::getRegionDistanceFromLeftEdgeOfFrame() const {
    return regionDistanceFromLeftEdgeOfFrame_;
}

void Addroi::setRegionDistanceFromTopEdgeOfFrame(const std::string& value) {
    regionDistanceFromTopEdgeOfFrame_ = value;
}

std::string Addroi::getRegionDistanceFromTopEdgeOfFrame() const {
    return regionDistanceFromTopEdgeOfFrame_;
}

void Addroi::setRegionWidth(const std::string& value) {
    regionWidth_ = value;
}

std::string Addroi::getRegionWidth() const {
    return regionWidth_;
}

void Addroi::setRegionHeight(const std::string& value) {
    regionHeight_ = value;
}

std::string Addroi::getRegionHeight() const {
    return regionHeight_;
}

void Addroi::setQoffset(const std::pair<int, int>& value) {
    qoffset_ = value;
}

std::pair<int, int> Addroi::getQoffset() const {
    return qoffset_;
}

void Addroi::setClear(bool value) {
    clear_ = value;
}

bool Addroi::getClear() const {
    return clear_;
}

std::string Addroi::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "addroi";

    bool first = true;

    if (regionDistanceFromLeftEdgeOfFrame_ != "0") {
        desc << (first ? "=" : ":") << "x=" << regionDistanceFromLeftEdgeOfFrame_;
        first = false;
    }
    if (regionDistanceFromTopEdgeOfFrame_ != "0") {
        desc << (first ? "=" : ":") << "y=" << regionDistanceFromTopEdgeOfFrame_;
        first = false;
    }
    if (regionWidth_ != "0") {
        desc << (first ? "=" : ":") << "w=" << regionWidth_;
        first = false;
    }
    if (regionHeight_ != "0") {
        desc << (first ? "=" : ":") << "h=" << regionHeight_;
        first = false;
    }
    if (qoffset_.first != 0 || qoffset_.second != 1) {
        desc << (first ? "=" : ":") << "qoffset=" << qoffset_.first << "/" << qoffset_.second;
        first = false;
    }
    if (clear_ != false) {
        desc << (first ? "=" : ":") << "clear=" << (clear_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
