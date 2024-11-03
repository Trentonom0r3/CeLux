#include "Setrange.hpp"
#include <sstream>

Setrange::Setrange(int range) {
    // Initialize member variables from parameters
    this->range_ = range;
}

Setrange::~Setrange() {
    // Destructor implementation (if needed)
}

void Setrange::setRange(int value) {
    range_ = value;
}

int Setrange::getRange() const {
    return range_;
}

std::string Setrange::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "setrange";

    bool first = true;

    if (range_ != -1) {
        desc << (first ? "=" : ":") << "range=" << range_;
        first = false;
    }

    return desc.str();
}
