#include "Telecine.hpp"
#include <sstream>

Telecine::Telecine(int first_field, const std::string& pattern) {
    // Initialize member variables from parameters
    this->first_field_ = first_field;
    this->pattern_ = pattern;
}

Telecine::~Telecine() {
    // Destructor implementation (if needed)
}

void Telecine::setFirst_field(int value) {
    first_field_ = value;
}

int Telecine::getFirst_field() const {
    return first_field_;
}

void Telecine::setPattern(const std::string& value) {
    pattern_ = value;
}

std::string Telecine::getPattern() const {
    return pattern_;
}

std::string Telecine::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "telecine";

    bool first = true;

    if (first_field_ != 0) {
        desc << (first ? "=" : ":") << "first_field=" << first_field_;
        first = false;
    }
    if (pattern_ != "23") {
        desc << (first ? "=" : ":") << "pattern=" << pattern_;
        first = false;
    }

    return desc.str();
}
