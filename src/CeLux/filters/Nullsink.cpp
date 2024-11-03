#include "Nullsink.hpp"
#include <sstream>

Nullsink::Nullsink() {
    // Initialize member variables with default values
}

Nullsink::~Nullsink() {
    // Destructor implementation (if needed)
}

std::string Nullsink::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "nullsink";

    bool first = true;


    return desc.str();
}
