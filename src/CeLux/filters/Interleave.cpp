#include "Interleave.hpp"
#include <sstream>

Interleave::Interleave(int nb_inputs, int duration) {
    // Initialize member variables from parameters
    this->nb_inputs_ = nb_inputs;
    this->duration_ = duration;
}

Interleave::~Interleave() {
    // Destructor implementation (if needed)
}

void Interleave::setNb_inputs(int value) {
    nb_inputs_ = value;
}

int Interleave::getNb_inputs() const {
    return nb_inputs_;
}

void Interleave::setDuration(int value) {
    duration_ = value;
}

int Interleave::getDuration() const {
    return duration_;
}

std::string Interleave::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "interleave";

    bool first = true;

    if (nb_inputs_ != 2) {
        desc << (first ? "=" : ":") << "nb_inputs=" << nb_inputs_;
        first = false;
    }
    if (duration_ != 0) {
        desc << (first ? "=" : ":") << "duration=" << duration_;
        first = false;
    }

    return desc.str();
}
