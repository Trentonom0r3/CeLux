#include "Hstack.hpp"
#include <sstream>

Hstack::Hstack(int inputs, bool shortest) {
    // Initialize member variables from parameters
    this->inputs_ = inputs;
    this->shortest_ = shortest;
}

Hstack::~Hstack() {
    // Destructor implementation (if needed)
}

void Hstack::setInputs(int value) {
    inputs_ = value;
}

int Hstack::getInputs() const {
    return inputs_;
}

void Hstack::setShortest(bool value) {
    shortest_ = value;
}

bool Hstack::getShortest() const {
    return shortest_;
}

std::string Hstack::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hstack";

    bool first = true;

    if (inputs_ != 2) {
        desc << (first ? "=" : ":") << "inputs=" << inputs_;
        first = false;
    }
    if (shortest_ != false) {
        desc << (first ? "=" : ":") << "shortest=" << (shortest_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
