#include "Anullsink.hpp"
#include <sstream>

Anullsink::Anullsink() {
    // Initialize member variables with default values
}

Anullsink::~Anullsink() {
    // Destructor implementation (if needed)
}

std::string Anullsink::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "anullsink";

    bool first = true;


    return desc.str();
}
