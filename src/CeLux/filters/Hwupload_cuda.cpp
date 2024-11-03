#include "Hwupload_cuda.hpp"
#include <sstream>

Hwupload_cuda::Hwupload_cuda(int device) {
    // Initialize member variables from parameters
    this->device_ = device;
}

Hwupload_cuda::~Hwupload_cuda() {
    // Destructor implementation (if needed)
}

void Hwupload_cuda::setDevice(int value) {
    device_ = value;
}

int Hwupload_cuda::getDevice() const {
    return device_;
}

std::string Hwupload_cuda::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hwupload_cuda";

    bool first = true;

    if (device_ != 0) {
        desc << (first ? "=" : ":") << "device=" << device_;
        first = false;
    }

    return desc.str();
}
