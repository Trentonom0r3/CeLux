#include "Weave.hpp"
#include <sstream>

Weave::Weave(int first_field) {
    // Initialize member variables from parameters
    this->first_field_ = first_field;
}

Weave::~Weave() {
    // Destructor implementation (if needed)
}

void Weave::setFirst_field(int value) {
    first_field_ = value;
}

int Weave::getFirst_field() const {
    return first_field_;
}

std::string Weave::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "weave";

    bool first = true;

    if (first_field_ != 0) {
        desc << (first ? "=" : ":") << "first_field=" << first_field_;
        first = false;
    }

    return desc.str();
}
