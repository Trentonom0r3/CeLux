#include "Rgbashift.hpp"
#include <sstream>

Rgbashift::Rgbashift(int rh, int rv, int gh, int gv, int bh, int bv, int ah, int av, int edge) {
    // Initialize member variables from parameters
    this->rh_ = rh;
    this->rv_ = rv;
    this->gh_ = gh;
    this->gv_ = gv;
    this->bh_ = bh;
    this->bv_ = bv;
    this->ah_ = ah;
    this->av_ = av;
    this->edge_ = edge;
}

Rgbashift::~Rgbashift() {
    // Destructor implementation (if needed)
}

void Rgbashift::setRh(int value) {
    rh_ = value;
}

int Rgbashift::getRh() const {
    return rh_;
}

void Rgbashift::setRv(int value) {
    rv_ = value;
}

int Rgbashift::getRv() const {
    return rv_;
}

void Rgbashift::setGh(int value) {
    gh_ = value;
}

int Rgbashift::getGh() const {
    return gh_;
}

void Rgbashift::setGv(int value) {
    gv_ = value;
}

int Rgbashift::getGv() const {
    return gv_;
}

void Rgbashift::setBh(int value) {
    bh_ = value;
}

int Rgbashift::getBh() const {
    return bh_;
}

void Rgbashift::setBv(int value) {
    bv_ = value;
}

int Rgbashift::getBv() const {
    return bv_;
}

void Rgbashift::setAh(int value) {
    ah_ = value;
}

int Rgbashift::getAh() const {
    return ah_;
}

void Rgbashift::setAv(int value) {
    av_ = value;
}

int Rgbashift::getAv() const {
    return av_;
}

void Rgbashift::setEdge(int value) {
    edge_ = value;
}

int Rgbashift::getEdge() const {
    return edge_;
}

std::string Rgbashift::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "rgbashift";

    bool first = true;

    if (rh_ != 0) {
        desc << (first ? "=" : ":") << "rh=" << rh_;
        first = false;
    }
    if (rv_ != 0) {
        desc << (first ? "=" : ":") << "rv=" << rv_;
        first = false;
    }
    if (gh_ != 0) {
        desc << (first ? "=" : ":") << "gh=" << gh_;
        first = false;
    }
    if (gv_ != 0) {
        desc << (first ? "=" : ":") << "gv=" << gv_;
        first = false;
    }
    if (bh_ != 0) {
        desc << (first ? "=" : ":") << "bh=" << bh_;
        first = false;
    }
    if (bv_ != 0) {
        desc << (first ? "=" : ":") << "bv=" << bv_;
        first = false;
    }
    if (ah_ != 0) {
        desc << (first ? "=" : ":") << "ah=" << ah_;
        first = false;
    }
    if (av_ != 0) {
        desc << (first ? "=" : ":") << "av=" << av_;
        first = false;
    }
    if (edge_ != 0) {
        desc << (first ? "=" : ":") << "edge=" << edge_;
        first = false;
    }

    return desc.str();
}
