#include "Nlmeans.hpp"
#include <sstream>

Nlmeans::Nlmeans(double denoisingStrength, int patchSize, int pc, int researchWindow, int rc) {
    // Initialize member variables from parameters
    this->denoisingStrength_ = denoisingStrength;
    this->patchSize_ = patchSize;
    this->pc_ = pc;
    this->researchWindow_ = researchWindow;
    this->rc_ = rc;
}

Nlmeans::~Nlmeans() {
    // Destructor implementation (if needed)
}

void Nlmeans::setDenoisingStrength(double value) {
    denoisingStrength_ = value;
}

double Nlmeans::getDenoisingStrength() const {
    return denoisingStrength_;
}

void Nlmeans::setPatchSize(int value) {
    patchSize_ = value;
}

int Nlmeans::getPatchSize() const {
    return patchSize_;
}

void Nlmeans::setPc(int value) {
    pc_ = value;
}

int Nlmeans::getPc() const {
    return pc_;
}

void Nlmeans::setResearchWindow(int value) {
    researchWindow_ = value;
}

int Nlmeans::getResearchWindow() const {
    return researchWindow_;
}

void Nlmeans::setRc(int value) {
    rc_ = value;
}

int Nlmeans::getRc() const {
    return rc_;
}

std::string Nlmeans::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "nlmeans";

    bool first = true;

    if (denoisingStrength_ != 1.00) {
        desc << (first ? "=" : ":") << "s=" << denoisingStrength_;
        first = false;
    }
    if (patchSize_ != 7) {
        desc << (first ? "=" : ":") << "p=" << patchSize_;
        first = false;
    }
    if (pc_ != 0) {
        desc << (first ? "=" : ":") << "pc=" << pc_;
        first = false;
    }
    if (researchWindow_ != 15) {
        desc << (first ? "=" : ":") << "r=" << researchWindow_;
        first = false;
    }
    if (rc_ != 0) {
        desc << (first ? "=" : ":") << "rc=" << rc_;
        first = false;
    }

    return desc.str();
}
