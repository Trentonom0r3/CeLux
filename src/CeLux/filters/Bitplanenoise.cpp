#include "Bitplanenoise.hpp"
#include <sstream>

Bitplanenoise::Bitplanenoise(int bitplane, bool filter) {
    // Initialize member variables from parameters
    this->bitplane_ = bitplane;
    this->filter_ = filter;
}

Bitplanenoise::~Bitplanenoise() {
    // Destructor implementation (if needed)
}

void Bitplanenoise::setBitplane(int value) {
    bitplane_ = value;
}

int Bitplanenoise::getBitplane() const {
    return bitplane_;
}

void Bitplanenoise::setFilter(bool value) {
    filter_ = value;
}

bool Bitplanenoise::getFilter() const {
    return filter_;
}

std::string Bitplanenoise::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "bitplanenoise";

    bool first = true;

    if (bitplane_ != 1) {
        desc << (first ? "=" : ":") << "bitplane=" << bitplane_;
        first = false;
    }
    if (filter_ != false) {
        desc << (first ? "=" : ":") << "filter=" << (filter_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
