#include "Doubleweave.hpp"
#include <sstream>

Doubleweave::Doubleweave(int first_field) {
    // Initialize member variables from parameters
    this->first_field_ = first_field;
}

Doubleweave::~Doubleweave() {
    // Destructor implementation (if needed)
}

void Doubleweave::setFirst_field(int value) {
    first_field_ = value;
}

int Doubleweave::getFirst_field() const {
    return first_field_;
}

std::string Doubleweave::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "doubleweave";

    bool first = true;

    if (first_field_ != 0) {
        desc << (first ? "=" : ":") << "first_field=" << first_field_;
        first = false;
    }

    return desc.str();
}
