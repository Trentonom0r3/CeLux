#include "Aphasemeter.hpp"
#include <sstream>

Aphasemeter::Aphasemeter(std::pair<int, int> rate, std::pair<int, int> size, int rc, int gc, int bc, const std::string& mpc, bool video, bool phasing, float tolerance, float angle, int64_t duration) {
    // Initialize member variables from parameters
    this->rate_ = rate;
    this->size_ = size;
    this->rc_ = rc;
    this->gc_ = gc;
    this->bc_ = bc;
    this->mpc_ = mpc;
    this->video_ = video;
    this->phasing_ = phasing;
    this->tolerance_ = tolerance;
    this->angle_ = angle;
    this->duration_ = duration;
}

Aphasemeter::~Aphasemeter() {
    // Destructor implementation (if needed)
}

void Aphasemeter::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Aphasemeter::getRate() const {
    return rate_;
}

void Aphasemeter::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Aphasemeter::getSize() const {
    return size_;
}

void Aphasemeter::setRc(int value) {
    rc_ = value;
}

int Aphasemeter::getRc() const {
    return rc_;
}

void Aphasemeter::setGc(int value) {
    gc_ = value;
}

int Aphasemeter::getGc() const {
    return gc_;
}

void Aphasemeter::setBc(int value) {
    bc_ = value;
}

int Aphasemeter::getBc() const {
    return bc_;
}

void Aphasemeter::setMpc(const std::string& value) {
    mpc_ = value;
}

std::string Aphasemeter::getMpc() const {
    return mpc_;
}

void Aphasemeter::setVideo(bool value) {
    video_ = value;
}

bool Aphasemeter::getVideo() const {
    return video_;
}

void Aphasemeter::setPhasing(bool value) {
    phasing_ = value;
}

bool Aphasemeter::getPhasing() const {
    return phasing_;
}

void Aphasemeter::setTolerance(float value) {
    tolerance_ = value;
}

float Aphasemeter::getTolerance() const {
    return tolerance_;
}

void Aphasemeter::setAngle(float value) {
    angle_ = value;
}

float Aphasemeter::getAngle() const {
    return angle_;
}

void Aphasemeter::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Aphasemeter::getDuration() const {
    return duration_;
}

std::string Aphasemeter::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "aphasemeter";

    bool first = true;

    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rc_ != 2) {
        desc << (first ? "=" : ":") << "rc=" << rc_;
        first = false;
    }
    if (gc_ != 7) {
        desc << (first ? "=" : ":") << "gc=" << gc_;
        first = false;
    }
    if (bc_ != 1) {
        desc << (first ? "=" : ":") << "bc=" << bc_;
        first = false;
    }
    if (mpc_ != "none") {
        desc << (first ? "=" : ":") << "mpc=" << mpc_;
        first = false;
    }
    if (video_ != true) {
        desc << (first ? "=" : ":") << "video=" << (video_ ? "1" : "0");
        first = false;
    }
    if (phasing_ != false) {
        desc << (first ? "=" : ":") << "phasing=" << (phasing_ ? "1" : "0");
        first = false;
    }
    if (tolerance_ != 0.00) {
        desc << (first ? "=" : ":") << "tolerance=" << tolerance_;
        first = false;
    }
    if (angle_ != 170.00) {
        desc << (first ? "=" : ":") << "angle=" << angle_;
        first = false;
    }
    if (duration_ != 2000000ULL) {
        desc << (first ? "=" : ":") << "duration=" << duration_;
        first = false;
    }

    return desc.str();
}
