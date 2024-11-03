#include "Dedot.hpp"
#include <sstream>

Dedot::Dedot(int filteringMode, float lt, float tl, float tc, float ct) {
    // Initialize member variables from parameters
    this->filteringMode_ = filteringMode;
    this->lt_ = lt;
    this->tl_ = tl;
    this->tc_ = tc;
    this->ct_ = ct;
}

Dedot::~Dedot() {
    // Destructor implementation (if needed)
}

void Dedot::setFilteringMode(int value) {
    filteringMode_ = value;
}

int Dedot::getFilteringMode() const {
    return filteringMode_;
}

void Dedot::setLt(float value) {
    lt_ = value;
}

float Dedot::getLt() const {
    return lt_;
}

void Dedot::setTl(float value) {
    tl_ = value;
}

float Dedot::getTl() const {
    return tl_;
}

void Dedot::setTc(float value) {
    tc_ = value;
}

float Dedot::getTc() const {
    return tc_;
}

void Dedot::setCt(float value) {
    ct_ = value;
}

float Dedot::getCt() const {
    return ct_;
}

std::string Dedot::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "dedot";

    bool first = true;

    if (filteringMode_ != 3) {
        desc << (first ? "=" : ":") << "m=" << filteringMode_;
        first = false;
    }
    if (lt_ != 0.08) {
        desc << (first ? "=" : ":") << "lt=" << lt_;
        first = false;
    }
    if (tl_ != 0.08) {
        desc << (first ? "=" : ":") << "tl=" << tl_;
        first = false;
    }
    if (tc_ != 0.06) {
        desc << (first ? "=" : ":") << "tc=" << tc_;
        first = false;
    }
    if (ct_ != 0.02) {
        desc << (first ? "=" : ":") << "ct=" << ct_;
        first = false;
    }

    return desc.str();
}
