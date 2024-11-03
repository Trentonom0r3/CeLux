#include "Bm3d.hpp"
#include <sstream>

Bm3d::Bm3d(float sigma, int block, int bstep, int group, int range, int mstep, float thmse, float hdthr, int estim, bool ref, int planes) {
    // Initialize member variables from parameters
    this->sigma_ = sigma;
    this->block_ = block;
    this->bstep_ = bstep;
    this->group_ = group;
    this->range_ = range;
    this->mstep_ = mstep;
    this->thmse_ = thmse;
    this->hdthr_ = hdthr;
    this->estim_ = estim;
    this->ref_ = ref;
    this->planes_ = planes;
}

Bm3d::~Bm3d() {
    // Destructor implementation (if needed)
}

void Bm3d::setSigma(float value) {
    sigma_ = value;
}

float Bm3d::getSigma() const {
    return sigma_;
}

void Bm3d::setBlock(int value) {
    block_ = value;
}

int Bm3d::getBlock() const {
    return block_;
}

void Bm3d::setBstep(int value) {
    bstep_ = value;
}

int Bm3d::getBstep() const {
    return bstep_;
}

void Bm3d::setGroup(int value) {
    group_ = value;
}

int Bm3d::getGroup() const {
    return group_;
}

void Bm3d::setRange(int value) {
    range_ = value;
}

int Bm3d::getRange() const {
    return range_;
}

void Bm3d::setMstep(int value) {
    mstep_ = value;
}

int Bm3d::getMstep() const {
    return mstep_;
}

void Bm3d::setThmse(float value) {
    thmse_ = value;
}

float Bm3d::getThmse() const {
    return thmse_;
}

void Bm3d::setHdthr(float value) {
    hdthr_ = value;
}

float Bm3d::getHdthr() const {
    return hdthr_;
}

void Bm3d::setEstim(int value) {
    estim_ = value;
}

int Bm3d::getEstim() const {
    return estim_;
}

void Bm3d::setRef(bool value) {
    ref_ = value;
}

bool Bm3d::getRef() const {
    return ref_;
}

void Bm3d::setPlanes(int value) {
    planes_ = value;
}

int Bm3d::getPlanes() const {
    return planes_;
}

std::string Bm3d::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "bm3d";

    bool first = true;

    if (sigma_ != 1.00) {
        desc << (first ? "=" : ":") << "sigma=" << sigma_;
        first = false;
    }
    if (block_ != 16) {
        desc << (first ? "=" : ":") << "block=" << block_;
        first = false;
    }
    if (bstep_ != 4) {
        desc << (first ? "=" : ":") << "bstep=" << bstep_;
        first = false;
    }
    if (group_ != 1) {
        desc << (first ? "=" : ":") << "group=" << group_;
        first = false;
    }
    if (range_ != 9) {
        desc << (first ? "=" : ":") << "range=" << range_;
        first = false;
    }
    if (mstep_ != 1) {
        desc << (first ? "=" : ":") << "mstep=" << mstep_;
        first = false;
    }
    if (thmse_ != 0.00) {
        desc << (first ? "=" : ":") << "thmse=" << thmse_;
        first = false;
    }
    if (hdthr_ != 2.70) {
        desc << (first ? "=" : ":") << "hdthr=" << hdthr_;
        first = false;
    }
    if (estim_ != 0) {
        desc << (first ? "=" : ":") << "estim=" << estim_;
        first = false;
    }
    if (ref_ != false) {
        desc << (first ? "=" : ":") << "ref=" << (ref_ ? "1" : "0");
        first = false;
    }
    if (planes_ != 7) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
