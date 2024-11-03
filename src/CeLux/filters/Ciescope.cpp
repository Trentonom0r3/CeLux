#include "Ciescope.hpp"
#include <sstream>

Ciescope::Ciescope(int system, int cie, int gamuts, int size, float intensity, float contrast, bool corrgamma, bool showwhite, double gamma, bool fill) {
    // Initialize member variables from parameters
    this->system_ = system;
    this->cie_ = cie;
    this->gamuts_ = gamuts;
    this->size_ = size;
    this->intensity_ = intensity;
    this->contrast_ = contrast;
    this->corrgamma_ = corrgamma;
    this->showwhite_ = showwhite;
    this->gamma_ = gamma;
    this->fill_ = fill;
}

Ciescope::~Ciescope() {
    // Destructor implementation (if needed)
}

void Ciescope::setSystem(int value) {
    system_ = value;
}

int Ciescope::getSystem() const {
    return system_;
}

void Ciescope::setCie(int value) {
    cie_ = value;
}

int Ciescope::getCie() const {
    return cie_;
}

void Ciescope::setGamuts(int value) {
    gamuts_ = value;
}

int Ciescope::getGamuts() const {
    return gamuts_;
}

void Ciescope::setSize(int value) {
    size_ = value;
}

int Ciescope::getSize() const {
    return size_;
}

void Ciescope::setIntensity(float value) {
    intensity_ = value;
}

float Ciescope::getIntensity() const {
    return intensity_;
}

void Ciescope::setContrast(float value) {
    contrast_ = value;
}

float Ciescope::getContrast() const {
    return contrast_;
}

void Ciescope::setCorrgamma(bool value) {
    corrgamma_ = value;
}

bool Ciescope::getCorrgamma() const {
    return corrgamma_;
}

void Ciescope::setShowwhite(bool value) {
    showwhite_ = value;
}

bool Ciescope::getShowwhite() const {
    return showwhite_;
}

void Ciescope::setGamma(double value) {
    gamma_ = value;
}

double Ciescope::getGamma() const {
    return gamma_;
}

void Ciescope::setFill(bool value) {
    fill_ = value;
}

bool Ciescope::getFill() const {
    return fill_;
}

std::string Ciescope::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "ciescope";

    bool first = true;

    if (system_ != 7) {
        desc << (first ? "=" : ":") << "system=" << system_;
        first = false;
    }
    if (cie_ != 0) {
        desc << (first ? "=" : ":") << "cie=" << cie_;
        first = false;
    }
    if (gamuts_ != 0) {
        desc << (first ? "=" : ":") << "gamuts=" << gamuts_;
        first = false;
    }
    if (size_ != 512) {
        desc << (first ? "=" : ":") << "size=" << size_;
        first = false;
    }
    if (intensity_ != 0.00) {
        desc << (first ? "=" : ":") << "intensity=" << intensity_;
        first = false;
    }
    if (contrast_ != 0.75) {
        desc << (first ? "=" : ":") << "contrast=" << contrast_;
        first = false;
    }
    if (corrgamma_ != true) {
        desc << (first ? "=" : ":") << "corrgamma=" << (corrgamma_ ? "1" : "0");
        first = false;
    }
    if (showwhite_ != false) {
        desc << (first ? "=" : ":") << "showwhite=" << (showwhite_ ? "1" : "0");
        first = false;
    }
    if (gamma_ != 2.60) {
        desc << (first ? "=" : ":") << "gamma=" << gamma_;
        first = false;
    }
    if (fill_ != true) {
        desc << (first ? "=" : ":") << "fill=" << (fill_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
