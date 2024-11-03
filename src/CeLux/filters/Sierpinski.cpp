#include "Sierpinski.hpp"
#include <sstream>

Sierpinski::Sierpinski(std::pair<int, int> size, std::pair<int, int> rate, int64_t seed, int jump, int type) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->seed_ = seed;
    this->jump_ = jump;
    this->type_ = type;
}

Sierpinski::~Sierpinski() {
    // Destructor implementation (if needed)
}

void Sierpinski::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Sierpinski::getSize() const {
    return size_;
}

void Sierpinski::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Sierpinski::getRate() const {
    return rate_;
}

void Sierpinski::setSeed(int64_t value) {
    seed_ = value;
}

int64_t Sierpinski::getSeed() const {
    return seed_;
}

void Sierpinski::setJump(int value) {
    jump_ = value;
}

int Sierpinski::getJump() const {
    return jump_;
}

void Sierpinski::setType(int value) {
    type_ = value;
}

int Sierpinski::getType() const {
    return type_;
}

std::string Sierpinski::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "sierpinski";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (seed_ != 0) {
        desc << (first ? "=" : ":") << "seed=" << seed_;
        first = false;
    }
    if (jump_ != 100) {
        desc << (first ? "=" : ":") << "jump=" << jump_;
        first = false;
    }
    if (type_ != 0) {
        desc << (first ? "=" : ":") << "type=" << type_;
        first = false;
    }

    return desc.str();
}
