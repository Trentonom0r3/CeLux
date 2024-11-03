#include "Dejudder.hpp"
#include <sstream>

Dejudder::Dejudder(int cycle) {
    // Initialize member variables from parameters
    this->cycle_ = cycle;
}

Dejudder::~Dejudder() {
    // Destructor implementation (if needed)
}

void Dejudder::setCycle(int value) {
    cycle_ = value;
}

int Dejudder::getCycle() const {
    return cycle_;
}

std::string Dejudder::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "dejudder";

    bool first = true;

    if (cycle_ != 4) {
        desc << (first ? "=" : ":") << "cycle=" << cycle_;
        first = false;
    }

    return desc.str();
}
