#include "Shuffleframes.hpp"
#include <sstream>

Shuffleframes::Shuffleframes(const std::string& mapping) {
    // Initialize member variables from parameters
    this->mapping_ = mapping;
}

Shuffleframes::~Shuffleframes() {
    // Destructor implementation (if needed)
}

void Shuffleframes::setMapping(const std::string& value) {
    mapping_ = value;
}

std::string Shuffleframes::getMapping() const {
    return mapping_;
}

std::string Shuffleframes::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "shuffleframes";

    bool first = true;

    if (mapping_ != "0") {
        desc << (first ? "=" : ":") << "mapping=" << mapping_;
        first = false;
    }

    return desc.str();
}
