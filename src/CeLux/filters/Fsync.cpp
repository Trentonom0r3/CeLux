#include "Fsync.hpp"
#include <sstream>

Fsync::Fsync(const std::string& file) {
    // Initialize member variables from parameters
    this->file_ = file;
}

Fsync::~Fsync() {
    // Destructor implementation (if needed)
}

void Fsync::setFile(const std::string& value) {
    file_ = value;
}

std::string Fsync::getFile() const {
    return file_;
}

std::string Fsync::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "fsync";

    bool first = true;

    if (!file_.empty()) {
        desc << (first ? "=" : ":") << "file=" << file_;
        first = false;
    }

    return desc.str();
}
