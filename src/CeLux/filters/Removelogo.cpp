#include "Removelogo.hpp"
#include <sstream>

Removelogo::Removelogo(const std::string& filename) {
    // Initialize member variables from parameters
    this->filename_ = filename;
}

Removelogo::~Removelogo() {
    // Destructor implementation (if needed)
}

void Removelogo::setFilename(const std::string& value) {
    filename_ = value;
}

std::string Removelogo::getFilename() const {
    return filename_;
}

std::string Removelogo::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "removelogo";

    bool first = true;

    if (!filename_.empty()) {
        desc << (first ? "=" : ":") << "filename=" << filename_;
        first = false;
    }

    return desc.str();
}
