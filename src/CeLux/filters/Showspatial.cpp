#include "Showspatial.hpp"
#include <sstream>

Showspatial::Showspatial(std::pair<int, int> size, int win_size, int win_func, std::pair<int, int> rate) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->win_size_ = win_size;
    this->win_func_ = win_func;
    this->rate_ = rate;
}

Showspatial::~Showspatial() {
    // Destructor implementation (if needed)
}

void Showspatial::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Showspatial::getSize() const {
    return size_;
}

void Showspatial::setWin_size(int value) {
    win_size_ = value;
}

int Showspatial::getWin_size() const {
    return win_size_;
}

void Showspatial::setWin_func(int value) {
    win_func_ = value;
}

int Showspatial::getWin_func() const {
    return win_func_;
}

void Showspatial::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Showspatial::getRate() const {
    return rate_;
}

std::string Showspatial::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showspatial";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (win_size_ != 4096) {
        desc << (first ? "=" : ":") << "win_size=" << win_size_;
        first = false;
    }
    if (win_func_ != 1) {
        desc << (first ? "=" : ":") << "win_func=" << win_func_;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }

    return desc.str();
}
