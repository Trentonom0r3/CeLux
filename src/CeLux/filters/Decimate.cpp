#include "Decimate.hpp"
#include <sstream>

Decimate::Decimate(int cycle, double dupthresh, double scthresh, int blockx, int blocky, bool ppsrc, bool chroma, bool mixed) {
    // Initialize member variables from parameters
    this->cycle_ = cycle;
    this->dupthresh_ = dupthresh;
    this->scthresh_ = scthresh;
    this->blockx_ = blockx;
    this->blocky_ = blocky;
    this->ppsrc_ = ppsrc;
    this->chroma_ = chroma;
    this->mixed_ = mixed;
}

Decimate::~Decimate() {
    // Destructor implementation (if needed)
}

void Decimate::setCycle(int value) {
    cycle_ = value;
}

int Decimate::getCycle() const {
    return cycle_;
}

void Decimate::setDupthresh(double value) {
    dupthresh_ = value;
}

double Decimate::getDupthresh() const {
    return dupthresh_;
}

void Decimate::setScthresh(double value) {
    scthresh_ = value;
}

double Decimate::getScthresh() const {
    return scthresh_;
}

void Decimate::setBlockx(int value) {
    blockx_ = value;
}

int Decimate::getBlockx() const {
    return blockx_;
}

void Decimate::setBlocky(int value) {
    blocky_ = value;
}

int Decimate::getBlocky() const {
    return blocky_;
}

void Decimate::setPpsrc(bool value) {
    ppsrc_ = value;
}

bool Decimate::getPpsrc() const {
    return ppsrc_;
}

void Decimate::setChroma(bool value) {
    chroma_ = value;
}

bool Decimate::getChroma() const {
    return chroma_;
}

void Decimate::setMixed(bool value) {
    mixed_ = value;
}

bool Decimate::getMixed() const {
    return mixed_;
}

std::string Decimate::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "decimate";

    bool first = true;

    if (cycle_ != 5) {
        desc << (first ? "=" : ":") << "cycle=" << cycle_;
        first = false;
    }
    if (dupthresh_ != 1.10) {
        desc << (first ? "=" : ":") << "dupthresh=" << dupthresh_;
        first = false;
    }
    if (scthresh_ != 15.00) {
        desc << (first ? "=" : ":") << "scthresh=" << scthresh_;
        first = false;
    }
    if (blockx_ != 32) {
        desc << (first ? "=" : ":") << "blockx=" << blockx_;
        first = false;
    }
    if (blocky_ != 32) {
        desc << (first ? "=" : ":") << "blocky=" << blocky_;
        first = false;
    }
    if (ppsrc_ != false) {
        desc << (first ? "=" : ":") << "ppsrc=" << (ppsrc_ ? "1" : "0");
        first = false;
    }
    if (chroma_ != true) {
        desc << (first ? "=" : ":") << "chroma=" << (chroma_ ? "1" : "0");
        first = false;
    }
    if (mixed_ != false) {
        desc << (first ? "=" : ":") << "mixed=" << (mixed_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
