#include "Deblock.hpp"
#include <sstream>

Deblock::Deblock(int filter, int block, float alpha, float beta, float gamma, float delta, int planes) {
    // Initialize member variables from parameters
    this->filter_ = filter;
    this->block_ = block;
    this->alpha_ = alpha;
    this->beta_ = beta;
    this->gamma_ = gamma;
    this->delta_ = delta;
    this->planes_ = planes;
}

Deblock::~Deblock() {
    // Destructor implementation (if needed)
}

void Deblock::setFilter(int value) {
    filter_ = value;
}

int Deblock::getFilter() const {
    return filter_;
}

void Deblock::setBlock(int value) {
    block_ = value;
}

int Deblock::getBlock() const {
    return block_;
}

void Deblock::setAlpha(float value) {
    alpha_ = value;
}

float Deblock::getAlpha() const {
    return alpha_;
}

void Deblock::setBeta(float value) {
    beta_ = value;
}

float Deblock::getBeta() const {
    return beta_;
}

void Deblock::setGamma(float value) {
    gamma_ = value;
}

float Deblock::getGamma() const {
    return gamma_;
}

void Deblock::setDelta(float value) {
    delta_ = value;
}

float Deblock::getDelta() const {
    return delta_;
}

void Deblock::setPlanes(int value) {
    planes_ = value;
}

int Deblock::getPlanes() const {
    return planes_;
}

std::string Deblock::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "deblock";

    bool first = true;

    if (filter_ != 1) {
        desc << (first ? "=" : ":") << "filter=" << filter_;
        first = false;
    }
    if (block_ != 8) {
        desc << (first ? "=" : ":") << "block=" << block_;
        first = false;
    }
    if (alpha_ != 0.10) {
        desc << (first ? "=" : ":") << "alpha=" << alpha_;
        first = false;
    }
    if (beta_ != 0.05) {
        desc << (first ? "=" : ":") << "beta=" << beta_;
        first = false;
    }
    if (gamma_ != 0.05) {
        desc << (first ? "=" : ":") << "gamma=" << gamma_;
        first = false;
    }
    if (delta_ != 0.05) {
        desc << (first ? "=" : ":") << "delta=" << delta_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
