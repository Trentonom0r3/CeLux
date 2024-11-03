#include "Readvitc.hpp"
#include <sstream>

Readvitc::Readvitc(int scan_max, double thr_b, double thr_w) {
    // Initialize member variables from parameters
    this->scan_max_ = scan_max;
    this->thr_b_ = thr_b;
    this->thr_w_ = thr_w;
}

Readvitc::~Readvitc() {
    // Destructor implementation (if needed)
}

void Readvitc::setScan_max(int value) {
    scan_max_ = value;
}

int Readvitc::getScan_max() const {
    return scan_max_;
}

void Readvitc::setThr_b(double value) {
    thr_b_ = value;
}

double Readvitc::getThr_b() const {
    return thr_b_;
}

void Readvitc::setThr_w(double value) {
    thr_w_ = value;
}

double Readvitc::getThr_w() const {
    return thr_w_;
}

std::string Readvitc::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "readvitc";

    bool first = true;

    if (scan_max_ != 45) {
        desc << (first ? "=" : ":") << "scan_max=" << scan_max_;
        first = false;
    }
    if (thr_b_ != 0.20) {
        desc << (first ? "=" : ":") << "thr_b=" << thr_b_;
        first = false;
    }
    if (thr_w_ != 0.60) {
        desc << (first ? "=" : ":") << "thr_w=" << thr_w_;
        first = false;
    }

    return desc.str();
}
