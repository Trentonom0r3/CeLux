#include "Mestimate.hpp"
#include <sstream>

Mestimate::Mestimate(int method, int mb_size, int search_param) {
    // Initialize member variables from parameters
    this->method_ = method;
    this->mb_size_ = mb_size;
    this->search_param_ = search_param;
}

Mestimate::~Mestimate() {
    // Destructor implementation (if needed)
}

void Mestimate::setMethod(int value) {
    method_ = value;
}

int Mestimate::getMethod() const {
    return method_;
}

void Mestimate::setMb_size(int value) {
    mb_size_ = value;
}

int Mestimate::getMb_size() const {
    return mb_size_;
}

void Mestimate::setSearch_param(int value) {
    search_param_ = value;
}

int Mestimate::getSearch_param() const {
    return search_param_;
}

std::string Mestimate::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "mestimate";

    bool first = true;

    if (method_ != 1) {
        desc << (first ? "=" : ":") << "method=" << method_;
        first = false;
    }
    if (mb_size_ != 16) {
        desc << (first ? "=" : ":") << "mb_size=" << mb_size_;
        first = false;
    }
    if (search_param_ != 7) {
        desc << (first ? "=" : ":") << "search_param=" << search_param_;
        first = false;
    }

    return desc.str();
}
