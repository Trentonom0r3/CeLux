#include "Zoneplate.hpp"
#include <sstream>

Zoneplate::Zoneplate(std::pair<int, int> size, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar, int precision, int xo, int yo, int to, int k0, int kx, int ky, int kt, int kxt, int kyt, int kxy, int kx2, int ky2, int kt2, int ku, int kv) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
    this->precision_ = precision;
    this->xo_ = xo;
    this->yo_ = yo;
    this->to_ = to;
    this->k0_ = k0;
    this->kx_ = kx;
    this->ky_ = ky;
    this->kt_ = kt;
    this->kxt_ = kxt;
    this->kyt_ = kyt;
    this->kxy_ = kxy;
    this->kx2_ = kx2;
    this->ky2_ = ky2;
    this->kt2_ = kt2;
    this->ku_ = ku;
    this->kv_ = kv;
}

Zoneplate::~Zoneplate() {
    // Destructor implementation (if needed)
}

void Zoneplate::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Zoneplate::getSize() const {
    return size_;
}

void Zoneplate::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Zoneplate::getRate() const {
    return rate_;
}

void Zoneplate::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Zoneplate::getDuration() const {
    return duration_;
}

void Zoneplate::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Zoneplate::getSar() const {
    return sar_;
}

void Zoneplate::setPrecision(int value) {
    precision_ = value;
}

int Zoneplate::getPrecision() const {
    return precision_;
}

void Zoneplate::setXo(int value) {
    xo_ = value;
}

int Zoneplate::getXo() const {
    return xo_;
}

void Zoneplate::setYo(int value) {
    yo_ = value;
}

int Zoneplate::getYo() const {
    return yo_;
}

void Zoneplate::setTo(int value) {
    to_ = value;
}

int Zoneplate::getTo() const {
    return to_;
}

void Zoneplate::setK0(int value) {
    k0_ = value;
}

int Zoneplate::getK0() const {
    return k0_;
}

void Zoneplate::setKx(int value) {
    kx_ = value;
}

int Zoneplate::getKx() const {
    return kx_;
}

void Zoneplate::setKy(int value) {
    ky_ = value;
}

int Zoneplate::getKy() const {
    return ky_;
}

void Zoneplate::setKt(int value) {
    kt_ = value;
}

int Zoneplate::getKt() const {
    return kt_;
}

void Zoneplate::setKxt(int value) {
    kxt_ = value;
}

int Zoneplate::getKxt() const {
    return kxt_;
}

void Zoneplate::setKyt(int value) {
    kyt_ = value;
}

int Zoneplate::getKyt() const {
    return kyt_;
}

void Zoneplate::setKxy(int value) {
    kxy_ = value;
}

int Zoneplate::getKxy() const {
    return kxy_;
}

void Zoneplate::setKx2(int value) {
    kx2_ = value;
}

int Zoneplate::getKx2() const {
    return kx2_;
}

void Zoneplate::setKy2(int value) {
    ky2_ = value;
}

int Zoneplate::getKy2() const {
    return ky2_;
}

void Zoneplate::setKt2(int value) {
    kt2_ = value;
}

int Zoneplate::getKt2() const {
    return kt2_;
}

void Zoneplate::setKu(int value) {
    ku_ = value;
}

int Zoneplate::getKu() const {
    return ku_;
}

void Zoneplate::setKv(int value) {
    kv_ = value;
}

int Zoneplate::getKv() const {
    return kv_;
}

std::string Zoneplate::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "zoneplate";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (duration_ != 0) {
        desc << (first ? "=" : ":") << "duration=" << duration_;
        first = false;
    }
    if (sar_.first != 0 || sar_.second != 1) {
        desc << (first ? "=" : ":") << "sar=" << sar_.first << "/" << sar_.second;
        first = false;
    }
    if (precision_ != 10) {
        desc << (first ? "=" : ":") << "precision=" << precision_;
        first = false;
    }
    if (xo_ != 0) {
        desc << (first ? "=" : ":") << "xo=" << xo_;
        first = false;
    }
    if (yo_ != 0) {
        desc << (first ? "=" : ":") << "yo=" << yo_;
        first = false;
    }
    if (to_ != 0) {
        desc << (first ? "=" : ":") << "to=" << to_;
        first = false;
    }
    if (k0_ != 0) {
        desc << (first ? "=" : ":") << "k0=" << k0_;
        first = false;
    }
    if (kx_ != 0) {
        desc << (first ? "=" : ":") << "kx=" << kx_;
        first = false;
    }
    if (ky_ != 0) {
        desc << (first ? "=" : ":") << "ky=" << ky_;
        first = false;
    }
    if (kt_ != 0) {
        desc << (first ? "=" : ":") << "kt=" << kt_;
        first = false;
    }
    if (kxt_ != 0) {
        desc << (first ? "=" : ":") << "kxt=" << kxt_;
        first = false;
    }
    if (kyt_ != 0) {
        desc << (first ? "=" : ":") << "kyt=" << kyt_;
        first = false;
    }
    if (kxy_ != 0) {
        desc << (first ? "=" : ":") << "kxy=" << kxy_;
        first = false;
    }
    if (kx2_ != 0) {
        desc << (first ? "=" : ":") << "kx2=" << kx2_;
        first = false;
    }
    if (ky2_ != 0) {
        desc << (first ? "=" : ":") << "ky2=" << ky2_;
        first = false;
    }
    if (kt2_ != 0) {
        desc << (first ? "=" : ":") << "kt2=" << kt2_;
        first = false;
    }
    if (ku_ != 0) {
        desc << (first ? "=" : ":") << "ku=" << ku_;
        first = false;
    }
    if (kv_ != 0) {
        desc << (first ? "=" : ":") << "kv=" << kv_;
        first = false;
    }

    return desc.str();
}
