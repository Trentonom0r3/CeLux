#include "Vstack.hpp"
#include <sstream>

Vstack::Vstack(int inputs, bool shortest) {
    // Initialize member variables from parameters
    this->inputs_ = inputs;
    this->shortest_ = shortest;
}

Vstack::~Vstack() {
    // Destructor implementation (if needed)
}

void Vstack::setInputs(int value) {
    inputs_ = value;
}

int Vstack::getInputs() const {
    return inputs_;
}

void Vstack::setShortest(bool value) {
    shortest_ = value;
}

bool Vstack::getShortest() const {
    return shortest_;
}

std::string Vstack::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "vstack";

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
