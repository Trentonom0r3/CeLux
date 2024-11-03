#include "Fieldmatch.hpp"
#include <sstream>

Fieldmatch::Fieldmatch(int order, int mode, bool ppsrc, int field, bool mchroma, int y1, double scthresh, int combmatch, int combdbg, int cthresh, bool chroma, int blockx, int blocky, int combpel) {
    // Initialize member variables from parameters
    this->order_ = order;
    this->mode_ = mode;
    this->ppsrc_ = ppsrc;
    this->field_ = field;
    this->mchroma_ = mchroma;
    this->y1_ = y1;
    this->scthresh_ = scthresh;
    this->combmatch_ = combmatch;
    this->combdbg_ = combdbg;
    this->cthresh_ = cthresh;
    this->chroma_ = chroma;
    this->blockx_ = blockx;
    this->blocky_ = blocky;
    this->combpel_ = combpel;
}

Fieldmatch::~Fieldmatch() {
    // Destructor implementation (if needed)
}

void Fieldmatch::setOrder(int value) {
    order_ = value;
}

int Fieldmatch::getOrder() const {
    return order_;
}

void Fieldmatch::setMode(int value) {
    mode_ = value;
}

int Fieldmatch::getMode() const {
    return mode_;
}

void Fieldmatch::setPpsrc(bool value) {
    ppsrc_ = value;
}

bool Fieldmatch::getPpsrc() const {
    return ppsrc_;
}

void Fieldmatch::setField(int value) {
    field_ = value;
}

int Fieldmatch::getField() const {
    return field_;
}

void Fieldmatch::setMchroma(bool value) {
    mchroma_ = value;
}

bool Fieldmatch::getMchroma() const {
    return mchroma_;
}

void Fieldmatch::setY1(int value) {
    y1_ = value;
}

int Fieldmatch::getY1() const {
    return y1_;
}

void Fieldmatch::setScthresh(double value) {
    scthresh_ = value;
}

double Fieldmatch::getScthresh() const {
    return scthresh_;
}

void Fieldmatch::setCombmatch(int value) {
    combmatch_ = value;
}

int Fieldmatch::getCombmatch() const {
    return combmatch_;
}

void Fieldmatch::setCombdbg(int value) {
    combdbg_ = value;
}

int Fieldmatch::getCombdbg() const {
    return combdbg_;
}

void Fieldmatch::setCthresh(int value) {
    cthresh_ = value;
}

int Fieldmatch::getCthresh() const {
    return cthresh_;
}

void Fieldmatch::setChroma(bool value) {
    chroma_ = value;
}

bool Fieldmatch::getChroma() const {
    return chroma_;
}

void Fieldmatch::setBlockx(int value) {
    blockx_ = value;
}

int Fieldmatch::getBlockx() const {
    return blockx_;
}

void Fieldmatch::setBlocky(int value) {
    blocky_ = value;
}

int Fieldmatch::getBlocky() const {
    return blocky_;
}

void Fieldmatch::setCombpel(int value) {
    combpel_ = value;
}

int Fieldmatch::getCombpel() const {
    return combpel_;
}

std::string Fieldmatch::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "fieldmatch";

    bool first = true;

    if (order_ != -1) {
        desc << (first ? "=" : ":") << "order=" << order_;
        first = false;
    }
    if (mode_ != 1) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (ppsrc_ != false) {
        desc << (first ? "=" : ":") << "ppsrc=" << (ppsrc_ ? "1" : "0");
        first = false;
    }
    if (field_ != -1) {
        desc << (first ? "=" : ":") << "field=" << field_;
        first = false;
    }
    if (mchroma_ != true) {
        desc << (first ? "=" : ":") << "mchroma=" << (mchroma_ ? "1" : "0");
        first = false;
    }
    if (y1_ != 0) {
        desc << (first ? "=" : ":") << "y1=" << y1_;
        first = false;
    }
    if (scthresh_ != 12.00) {
        desc << (first ? "=" : ":") << "scthresh=" << scthresh_;
        first = false;
    }
    if (combmatch_ != 1) {
        desc << (first ? "=" : ":") << "combmatch=" << combmatch_;
        first = false;
    }
    if (combdbg_ != 0) {
        desc << (first ? "=" : ":") << "combdbg=" << combdbg_;
        first = false;
    }
    if (cthresh_ != 9) {
        desc << (first ? "=" : ":") << "cthresh=" << cthresh_;
        first = false;
    }
    if (chroma_ != false) {
        desc << (first ? "=" : ":") << "chroma=" << (chroma_ ? "1" : "0");
        first = false;
    }
    if (blockx_ != 16) {
        desc << (first ? "=" : ":") << "blockx=" << blockx_;
        first = false;
    }
    if (blocky_ != 16) {
        desc << (first ? "=" : ":") << "blocky=" << blocky_;
        first = false;
    }
    if (combpel_ != 80) {
        desc << (first ? "=" : ":") << "combpel=" << combpel_;
        first = false;
    }

    return desc.str();
}
