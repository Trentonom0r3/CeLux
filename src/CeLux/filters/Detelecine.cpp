#include "Detelecine.hpp"
#include <sstream>

Detelecine::Detelecine(int first_field, const std::string& pattern, int start_frame) {
    // Initialize member variables from parameters
    this->first_field_ = first_field;
    this->pattern_ = pattern;
    this->start_frame_ = start_frame;
}

Detelecine::~Detelecine() {
    // Destructor implementation (if needed)
}

void Detelecine::setFirst_field(int value) {
    first_field_ = value;
}

int Detelecine::getFirst_field() const {
    return first_field_;
}

void Detelecine::setPattern(const std::string& value) {
    pattern_ = value;
}

std::string Detelecine::getPattern() const {
    return pattern_;
}

void Detelecine::setStart_frame(int value) {
    start_frame_ = value;
}

int Detelecine::getStart_frame() const {
    return start_frame_;
}

std::string Detelecine::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "detelecine";

    bool first = true;

    if (first_field_ != 0) {
        desc << (first ? "=" : ":") << "first_field=" << first_field_;
        first = false;
    }
    if (pattern_ != "23") {
        desc << (first ? "=" : ":") << "pattern=" << pattern_;
        first = false;
    }
    if (start_frame_ != 0) {
        desc << (first ? "=" : ":") << "start_frame=" << start_frame_;
        first = false;
    }

    return desc.str();
}
