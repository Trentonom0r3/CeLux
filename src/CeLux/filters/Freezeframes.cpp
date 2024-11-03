#include "Freezeframes.hpp"
#include <sstream>

Freezeframes::Freezeframes(int64_t first, int64_t last, int64_t replace) {
    // Initialize member variables from parameters
    this->first_ = first;
    this->last_ = last;
    this->replace_ = replace;
}

Freezeframes::~Freezeframes() {
    // Destructor implementation (if needed)
}

void Freezeframes::setFirst(int64_t value) {
    first_ = value;
}

int64_t Freezeframes::getFirst() const {
    return first_;
}

void Freezeframes::setLast(int64_t value) {
    last_ = value;
}

int64_t Freezeframes::getLast() const {
    return last_;
}

void Freezeframes::setReplace(int64_t value) {
    replace_ = value;
}

int64_t Freezeframes::getReplace() const {
    return replace_;
}

std::string Freezeframes::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "freezeframes";

    bool first = true;

    if (first_ != 0ULL) {
        desc << (first ? "=" : ":") << "first=" << first_;
        first = false;
    }
    if (last_ != 0ULL) {
        desc << (first ? "=" : ":") << "last=" << last_;
        first = false;
    }
    if (replace_ != 0ULL) {
        desc << (first ? "=" : ":") << "replace=" << replace_;
        first = false;
    }

    return desc.str();
}
