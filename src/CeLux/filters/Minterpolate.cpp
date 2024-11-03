#include "Minterpolate.hpp"
#include <sstream>

Minterpolate::Minterpolate(std::pair<int, int> fps, int mi_mode, int mc_mode, int me_mode, int me, int mb_size, int search_param, int vsbmc, int scd, double scd_threshold) {
    // Initialize member variables from parameters
    this->fps_ = fps;
    this->mi_mode_ = mi_mode;
    this->mc_mode_ = mc_mode;
    this->me_mode_ = me_mode;
    this->me_ = me;
    this->mb_size_ = mb_size;
    this->search_param_ = search_param;
    this->vsbmc_ = vsbmc;
    this->scd_ = scd;
    this->scd_threshold_ = scd_threshold;
}

Minterpolate::~Minterpolate() {
    // Destructor implementation (if needed)
}

void Minterpolate::setFps(const std::pair<int, int>& value) {
    fps_ = value;
}

std::pair<int, int> Minterpolate::getFps() const {
    return fps_;
}

void Minterpolate::setMi_mode(int value) {
    mi_mode_ = value;
}

int Minterpolate::getMi_mode() const {
    return mi_mode_;
}

void Minterpolate::setMc_mode(int value) {
    mc_mode_ = value;
}

int Minterpolate::getMc_mode() const {
    return mc_mode_;
}

void Minterpolate::setMe_mode(int value) {
    me_mode_ = value;
}

int Minterpolate::getMe_mode() const {
    return me_mode_;
}

void Minterpolate::setMe(int value) {
    me_ = value;
}

int Minterpolate::getMe() const {
    return me_;
}

void Minterpolate::setMb_size(int value) {
    mb_size_ = value;
}

int Minterpolate::getMb_size() const {
    return mb_size_;
}

void Minterpolate::setSearch_param(int value) {
    search_param_ = value;
}

int Minterpolate::getSearch_param() const {
    return search_param_;
}

void Minterpolate::setVsbmc(int value) {
    vsbmc_ = value;
}

int Minterpolate::getVsbmc() const {
    return vsbmc_;
}

void Minterpolate::setScd(int value) {
    scd_ = value;
}

int Minterpolate::getScd() const {
    return scd_;
}

void Minterpolate::setScd_threshold(double value) {
    scd_threshold_ = value;
}

double Minterpolate::getScd_threshold() const {
    return scd_threshold_;
}

std::string Minterpolate::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "minterpolate";

    bool first = true;

    if (fps_.first != 0 || fps_.second != 1) {
        desc << (first ? "=" : ":") << "fps=" << fps_.first << "/" << fps_.second;
        first = false;
    }
    if (mi_mode_ != 2) {
        desc << (first ? "=" : ":") << "mi_mode=" << mi_mode_;
        first = false;
    }
    if (mc_mode_ != 0) {
        desc << (first ? "=" : ":") << "mc_mode=" << mc_mode_;
        first = false;
    }
    if (me_mode_ != 1) {
        desc << (first ? "=" : ":") << "me_mode=" << me_mode_;
        first = false;
    }
    if (me_ != 8) {
        desc << (first ? "=" : ":") << "me=" << me_;
        first = false;
    }
    if (mb_size_ != 16) {
        desc << (first ? "=" : ":") << "mb_size=" << mb_size_;
        first = false;
    }
    if (search_param_ != 32) {
        desc << (first ? "=" : ":") << "search_param=" << search_param_;
        first = false;
    }
    if (vsbmc_ != 0) {
        desc << (first ? "=" : ":") << "vsbmc=" << vsbmc_;
        first = false;
    }
    if (scd_ != 1) {
        desc << (first ? "=" : ":") << "scd=" << scd_;
        first = false;
    }
    if (scd_threshold_ != 10.00) {
        desc << (first ? "=" : ":") << "scd_threshold=" << scd_threshold_;
        first = false;
    }

    return desc.str();
}
