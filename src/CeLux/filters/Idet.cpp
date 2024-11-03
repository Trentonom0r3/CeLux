#include "Idet.hpp"
#include <sstream>

Idet::Idet(float intl_thres, float prog_thres, float rep_thres, float half_life, int analyze_interlaced_flag) {
    // Initialize member variables from parameters
    this->intl_thres_ = intl_thres;
    this->prog_thres_ = prog_thres;
    this->rep_thres_ = rep_thres;
    this->half_life_ = half_life;
    this->analyze_interlaced_flag_ = analyze_interlaced_flag;
}

Idet::~Idet() {
    // Destructor implementation (if needed)
}

void Idet::setIntl_thres(float value) {
    intl_thres_ = value;
}

float Idet::getIntl_thres() const {
    return intl_thres_;
}

void Idet::setProg_thres(float value) {
    prog_thres_ = value;
}

float Idet::getProg_thres() const {
    return prog_thres_;
}

void Idet::setRep_thres(float value) {
    rep_thres_ = value;
}

float Idet::getRep_thres() const {
    return rep_thres_;
}

void Idet::setHalf_life(float value) {
    half_life_ = value;
}

float Idet::getHalf_life() const {
    return half_life_;
}

void Idet::setAnalyze_interlaced_flag(int value) {
    analyze_interlaced_flag_ = value;
}

int Idet::getAnalyze_interlaced_flag() const {
    return analyze_interlaced_flag_;
}

std::string Idet::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "idet";

    bool first = true;

    if (intl_thres_ != 1.04) {
        desc << (first ? "=" : ":") << "intl_thres=" << intl_thres_;
        first = false;
    }
    if (prog_thres_ != 1.50) {
        desc << (first ? "=" : ":") << "prog_thres=" << prog_thres_;
        first = false;
    }
    if (rep_thres_ != 3.00) {
        desc << (first ? "=" : ":") << "rep_thres=" << rep_thres_;
        first = false;
    }
    if (half_life_ != 0.00) {
        desc << (first ? "=" : ":") << "half_life=" << half_life_;
        first = false;
    }
    if (analyze_interlaced_flag_ != 0) {
        desc << (first ? "=" : ":") << "analyze_interlaced_flag=" << analyze_interlaced_flag_;
        first = false;
    }

    return desc.str();
}
