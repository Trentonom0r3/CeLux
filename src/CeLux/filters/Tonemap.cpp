#include "Tonemap.hpp"
#include <sstream>

Tonemap::Tonemap(int tonemap, double param, double desat, double peak) {
    // Initialize member variables from parameters
    this->tonemap_ = tonemap;
    this->param_ = param;
    this->desat_ = desat;
    this->peak_ = peak;
}

Tonemap::~Tonemap() {
    // Destructor implementation (if needed)
}

void Tonemap::setTonemap(int value) {
    tonemap_ = value;
}

int Tonemap::getTonemap() const {
    return tonemap_;
}

void Tonemap::setParam(double value) {
    param_ = value;
}

double Tonemap::getParam() const {
    return param_;
}

void Tonemap::setDesat(double value) {
    desat_ = value;
}

double Tonemap::getDesat() const {
    return desat_;
}

void Tonemap::setPeak(double value) {
    peak_ = value;
}

double Tonemap::getPeak() const {
    return peak_;
}

std::string Tonemap::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "tonemap";

    bool first = true;

    if (tonemap_ != 0) {
        desc << (first ? "=" : ":") << "tonemap=" << tonemap_;
        first = false;
    }
    if (param_ != 0) {
        desc << (first ? "=" : ":") << "param=" << param_;
        first = false;
    }
    if (desat_ != 2.00) {
        desc << (first ? "=" : ":") << "desat=" << desat_;
        first = false;
    }
    if (peak_ != 0.00) {
        desc << (first ? "=" : ":") << "peak=" << peak_;
        first = false;
    }

    return desc.str();
}
