#include "Framestep.hpp"
#include <sstream>

Framestep::Framestep(int step) {
    // Initialize member variables from parameters
    this->step_ = step;
}

Framestep::~Framestep() {
    // Destructor implementation (if needed)
}

void Framestep::setStep(int value) {
    step_ = value;
}

int Framestep::getStep() const {
    return step_;
}

std::string Framestep::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "framestep";

    bool first = true;

    if (step_ != 1) {
        desc << (first ? "=" : ":") << "step=" << step_;
        first = false;
    }

    return desc.str();
}
