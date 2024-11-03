#include "Colorchart.hpp"
#include <sstream>

Colorchart::Colorchart(std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar, std::pair<int, int> patch_size, int preset) {
    // Initialize member variables from parameters
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
    this->patch_size_ = patch_size;
    this->preset_ = preset;
}

Colorchart::~Colorchart() {
    // Destructor implementation (if needed)
}

void Colorchart::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Colorchart::getRate() const {
    return rate_;
}

void Colorchart::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Colorchart::getDuration() const {
    return duration_;
}

void Colorchart::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Colorchart::getSar() const {
    return sar_;
}

void Colorchart::setPatch_size(const std::pair<int, int>& value) {
    patch_size_ = value;
}

std::pair<int, int> Colorchart::getPatch_size() const {
    return patch_size_;
}

void Colorchart::setPreset(int value) {
    preset_ = value;
}

int Colorchart::getPreset() const {
    return preset_;
}

std::string Colorchart::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorchart";

    bool first = true;

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
    if (patch_size_.first != 0 || patch_size_.second != 1) {
        desc << (first ? "=" : ":") << "patch_size=" << patch_size_.first << "/" << patch_size_.second;
        first = false;
    }
    if (preset_ != 0) {
        desc << (first ? "=" : ":") << "preset=" << preset_;
        first = false;
    }

    return desc.str();
}
