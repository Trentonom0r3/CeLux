#include "Bbox.hpp"
#include <sstream>

Bbox::Bbox(int min_val) {
    // Initialize member variables from parameters
    this->min_val_ = min_val;
}

Bbox::~Bbox() {
    // Destructor implementation (if needed)
}

void Bbox::setMin_val(int value) {
    min_val_ = value;
}

int Bbox::getMin_val() const {
    return min_val_;
}

std::string Bbox::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "bbox";

    bool first = true;

    if (min_val_ != 16) {
        desc << (first ? "=" : ":") << "min_val=" << min_val_;
        first = false;
    }

    return desc.str();
}
