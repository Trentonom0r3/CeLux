#include "Fftdnoiz.hpp"
#include <sstream>

Fftdnoiz::Fftdnoiz(float sigma, float amount, int block, float overlap, int method, int prev, int next, int planes, int window) {
    // Initialize member variables from parameters
    this->sigma_ = sigma;
    this->amount_ = amount;
    this->block_ = block;
    this->overlap_ = overlap;
    this->method_ = method;
    this->prev_ = prev;
    this->next_ = next;
    this->planes_ = planes;
    this->window_ = window;
}

Fftdnoiz::~Fftdnoiz() {
    // Destructor implementation (if needed)
}

void Fftdnoiz::setSigma(float value) {
    sigma_ = value;
}

float Fftdnoiz::getSigma() const {
    return sigma_;
}

void Fftdnoiz::setAmount(float value) {
    amount_ = value;
}

float Fftdnoiz::getAmount() const {
    return amount_;
}

void Fftdnoiz::setBlock(int value) {
    block_ = value;
}

int Fftdnoiz::getBlock() const {
    return block_;
}

void Fftdnoiz::setOverlap(float value) {
    overlap_ = value;
}

float Fftdnoiz::getOverlap() const {
    return overlap_;
}

void Fftdnoiz::setMethod(int value) {
    method_ = value;
}

int Fftdnoiz::getMethod() const {
    return method_;
}

void Fftdnoiz::setPrev(int value) {
    prev_ = value;
}

int Fftdnoiz::getPrev() const {
    return prev_;
}

void Fftdnoiz::setNext(int value) {
    next_ = value;
}

int Fftdnoiz::getNext() const {
    return next_;
}

void Fftdnoiz::setPlanes(int value) {
    planes_ = value;
}

int Fftdnoiz::getPlanes() const {
    return planes_;
}

void Fftdnoiz::setWindow(int value) {
    window_ = value;
}

int Fftdnoiz::getWindow() const {
    return window_;
}

std::string Fftdnoiz::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "fftdnoiz";

    bool first = true;

    if (sigma_ != 1.00) {
        desc << (first ? "=" : ":") << "sigma=" << sigma_;
        first = false;
    }
    if (amount_ != 1.00) {
        desc << (first ? "=" : ":") << "amount=" << amount_;
        first = false;
    }
    if (block_ != 32) {
        desc << (first ? "=" : ":") << "block=" << block_;
        first = false;
    }
    if (overlap_ != 0.50) {
        desc << (first ? "=" : ":") << "overlap=" << overlap_;
        first = false;
    }
    if (method_ != 0) {
        desc << (first ? "=" : ":") << "method=" << method_;
        first = false;
    }
    if (prev_ != 0) {
        desc << (first ? "=" : ":") << "prev=" << prev_;
        first = false;
    }
    if (next_ != 0) {
        desc << (first ? "=" : ":") << "next=" << next_;
        first = false;
    }
    if (planes_ != 7) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (window_ != 1) {
        desc << (first ? "=" : ":") << "window=" << window_;
        first = false;
    }

    return desc.str();
}
