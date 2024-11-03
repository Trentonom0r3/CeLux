#include "Bench.hpp"
#include <sstream>

Bench::Bench(int action) {
    // Initialize member variables from parameters
    this->action_ = action;
}

Bench::~Bench() {
    // Destructor implementation (if needed)
}

void Bench::setAction(int value) {
    action_ = value;
}

int Bench::getAction() const {
    return action_;
}

std::string Bench::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "bench";

    bool first = true;

    if (action_ != 0) {
        desc << (first ? "=" : ":") << "action=" << action_;
        first = false;
    }

    return desc.str();
}
