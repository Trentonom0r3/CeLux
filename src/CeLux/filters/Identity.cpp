#include "Identity.hpp"
#include <sstream>

Identity::Identity() {
    // Initialize member variables with default values
}

Identity::~Identity() {
    // Destructor implementation (if needed)
}

std::string Identity::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "identity";

    bool first = true;


    return desc.str();
}
