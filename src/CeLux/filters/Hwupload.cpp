#include "Hwupload.hpp"
#include <sstream>

Hwupload::Hwupload(const std::string& derive_device) {
    // Initialize member variables from parameters
    this->derive_device_ = derive_device;
}

Hwupload::~Hwupload() {
    // Destructor implementation (if needed)
}

void Hwupload::setDerive_device(const std::string& value) {
    derive_device_ = value;
}

std::string Hwupload::getDerive_device() const {
    return derive_device_;
}

std::string Hwupload::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hwupload";

    bool first = true;

    if (!derive_device_.empty()) {
        desc << (first ? "=" : ":") << "derive_device=" << derive_device_;
        first = false;
    }

    return desc.str();
}
