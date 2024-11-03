#include "Colorbalance.hpp"
#include <sstream>

Colorbalance::Colorbalance(float rs, float gs, float bs, float rm, float gm, float bm, float rh, float gh, float bh, bool pl) {
    // Initialize member variables from parameters
    this->rs_ = rs;
    this->gs_ = gs;
    this->bs_ = bs;
    this->rm_ = rm;
    this->gm_ = gm;
    this->bm_ = bm;
    this->rh_ = rh;
    this->gh_ = gh;
    this->bh_ = bh;
    this->pl_ = pl;
}

Colorbalance::~Colorbalance() {
    // Destructor implementation (if needed)
}

void Colorbalance::setRs(float value) {
    rs_ = value;
}

float Colorbalance::getRs() const {
    return rs_;
}

void Colorbalance::setGs(float value) {
    gs_ = value;
}

float Colorbalance::getGs() const {
    return gs_;
}

void Colorbalance::setBs(float value) {
    bs_ = value;
}

float Colorbalance::getBs() const {
    return bs_;
}

void Colorbalance::setRm(float value) {
    rm_ = value;
}

float Colorbalance::getRm() const {
    return rm_;
}

void Colorbalance::setGm(float value) {
    gm_ = value;
}

float Colorbalance::getGm() const {
    return gm_;
}

void Colorbalance::setBm(float value) {
    bm_ = value;
}

float Colorbalance::getBm() const {
    return bm_;
}

void Colorbalance::setRh(float value) {
    rh_ = value;
}

float Colorbalance::getRh() const {
    return rh_;
}

void Colorbalance::setGh(float value) {
    gh_ = value;
}

float Colorbalance::getGh() const {
    return gh_;
}

void Colorbalance::setBh(float value) {
    bh_ = value;
}

float Colorbalance::getBh() const {
    return bh_;
}

void Colorbalance::setPl(bool value) {
    pl_ = value;
}

bool Colorbalance::getPl() const {
    return pl_;
}

std::string Colorbalance::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorbalance";

    bool first = true;

    if (rs_ != 0.00) {
        desc << (first ? "=" : ":") << "rs=" << rs_;
        first = false;
    }
    if (gs_ != 0.00) {
        desc << (first ? "=" : ":") << "gs=" << gs_;
        first = false;
    }
    if (bs_ != 0.00) {
        desc << (first ? "=" : ":") << "bs=" << bs_;
        first = false;
    }
    if (rm_ != 0.00) {
        desc << (first ? "=" : ":") << "rm=" << rm_;
        first = false;
    }
    if (gm_ != 0.00) {
        desc << (first ? "=" : ":") << "gm=" << gm_;
        first = false;
    }
    if (bm_ != 0.00) {
        desc << (first ? "=" : ":") << "bm=" << bm_;
        first = false;
    }
    if (rh_ != 0.00) {
        desc << (first ? "=" : ":") << "rh=" << rh_;
        first = false;
    }
    if (gh_ != 0.00) {
        desc << (first ? "=" : ":") << "gh=" << gh_;
        first = false;
    }
    if (bh_ != 0.00) {
        desc << (first ? "=" : ":") << "bh=" << bh_;
        first = false;
    }
    if (pl_ != false) {
        desc << (first ? "=" : ":") << "pl=" << (pl_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
