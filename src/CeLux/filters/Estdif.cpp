#include "Estdif.hpp"
#include <sstream>

Estdif::Estdif(int mode, int parity, int deint, int rslope, int redge, int ecost, int mcost, int dcost, int interp) {
    // Initialize member variables from parameters
    this->mode_ = mode;
    this->parity_ = parity;
    this->deint_ = deint;
    this->rslope_ = rslope;
    this->redge_ = redge;
    this->ecost_ = ecost;
    this->mcost_ = mcost;
    this->dcost_ = dcost;
    this->interp_ = interp;
}

Estdif::~Estdif() {
    // Destructor implementation (if needed)
}

void Estdif::setMode(int value) {
    mode_ = value;
}

int Estdif::getMode() const {
    return mode_;
}

void Estdif::setParity(int value) {
    parity_ = value;
}

int Estdif::getParity() const {
    return parity_;
}

void Estdif::setDeint(int value) {
    deint_ = value;
}

int Estdif::getDeint() const {
    return deint_;
}

void Estdif::setRslope(int value) {
    rslope_ = value;
}

int Estdif::getRslope() const {
    return rslope_;
}

void Estdif::setRedge(int value) {
    redge_ = value;
}

int Estdif::getRedge() const {
    return redge_;
}

void Estdif::setEcost(int value) {
    ecost_ = value;
}

int Estdif::getEcost() const {
    return ecost_;
}

void Estdif::setMcost(int value) {
    mcost_ = value;
}

int Estdif::getMcost() const {
    return mcost_;
}

void Estdif::setDcost(int value) {
    dcost_ = value;
}

int Estdif::getDcost() const {
    return dcost_;
}

void Estdif::setInterp(int value) {
    interp_ = value;
}

int Estdif::getInterp() const {
    return interp_;
}

std::string Estdif::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "estdif";

    bool first = true;

    if (mode_ != 1) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (parity_ != -1) {
        desc << (first ? "=" : ":") << "parity=" << parity_;
        first = false;
    }
    if (deint_ != 0) {
        desc << (first ? "=" : ":") << "deint=" << deint_;
        first = false;
    }
    if (rslope_ != 1) {
        desc << (first ? "=" : ":") << "rslope=" << rslope_;
        first = false;
    }
    if (redge_ != 2) {
        desc << (first ? "=" : ":") << "redge=" << redge_;
        first = false;
    }
    if (ecost_ != 2) {
        desc << (first ? "=" : ":") << "ecost=" << ecost_;
        first = false;
    }
    if (mcost_ != 1) {
        desc << (first ? "=" : ":") << "mcost=" << mcost_;
        first = false;
    }
    if (dcost_ != 1) {
        desc << (first ? "=" : ":") << "dcost=" << dcost_;
        first = false;
    }
    if (interp_ != 1) {
        desc << (first ? "=" : ":") << "interp=" << interp_;
        first = false;
    }

    return desc.str();
}
