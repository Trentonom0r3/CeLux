#include "Latency.hpp"
#include <sstream>

Latency::Latency() {
    // Initialize member variables with default values
}

Latency::~Latency() {
    // Destructor implementation (if needed)
}

std::string Latency::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "latency";

    bool first = true;


    return desc.str();
}
