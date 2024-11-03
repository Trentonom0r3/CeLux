#include "Chromashift.hpp"
#include <sstream>

Chromashift::Chromashift(int cbh, int cbv, int crh, int crv, int edge) {
    // Initialize member variables from parameters
    this->cbh_ = cbh;
    this->cbv_ = cbv;
    this->crh_ = crh;
    this->crv_ = crv;
    this->edge_ = edge;
}

Chromashift::~Chromashift() {
    // Destructor implementation (if needed)
}

void Chromashift::setCbh(int value) {
    cbh_ = value;
}

int Chromashift::getCbh() const {
    return cbh_;
}

void Chromashift::setCbv(int value) {
    cbv_ = value;
}

int Chromashift::getCbv() const {
    return cbv_;
}

void Chromashift::setCrh(int value) {
    crh_ = value;
}

int Chromashift::getCrh() const {
    return crh_;
}

void Chromashift::setCrv(int value) {
    crv_ = value;
}

int Chromashift::getCrv() const {
    return crv_;
}

void Chromashift::setEdge(int value) {
    edge_ = value;
}

int Chromashift::getEdge() const {
    return edge_;
}

std::string Chromashift::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "chromashift";

    bool first = true;

    if (cbh_ != 0) {
        desc << (first ? "=" : ":") << "cbh=" << cbh_;
        first = false;
    }
    if (cbv_ != 0) {
        desc << (first ? "=" : ":") << "cbv=" << cbv_;
        first = false;
    }
    if (crh_ != 0) {
        desc << (first ? "=" : ":") << "crh=" << crh_;
        first = false;
    }
    if (crv_ != 0) {
        desc << (first ? "=" : ":") << "crv=" << crv_;
        first = false;
    }
    if (edge_ != 0) {
        desc << (first ? "=" : ":") << "edge=" << edge_;
        first = false;
    }

    return desc.str();
}
