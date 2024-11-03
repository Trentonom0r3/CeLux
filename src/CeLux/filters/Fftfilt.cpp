#include "Fftfilt.hpp"
#include <sstream>

Fftfilt::Fftfilt(int dc_Y, int dc_U, int dc_V, const std::string& weight_Y, const std::string& weight_U, const std::string& weight_V, int eval) {
    // Initialize member variables from parameters
    this->dc_Y_ = dc_Y;
    this->dc_U_ = dc_U;
    this->dc_V_ = dc_V;
    this->weight_Y_ = weight_Y;
    this->weight_U_ = weight_U;
    this->weight_V_ = weight_V;
    this->eval_ = eval;
}

Fftfilt::~Fftfilt() {
    // Destructor implementation (if needed)
}

void Fftfilt::setDc_Y(int value) {
    dc_Y_ = value;
}

int Fftfilt::getDc_Y() const {
    return dc_Y_;
}

void Fftfilt::setDc_U(int value) {
    dc_U_ = value;
}

int Fftfilt::getDc_U() const {
    return dc_U_;
}

void Fftfilt::setDc_V(int value) {
    dc_V_ = value;
}

int Fftfilt::getDc_V() const {
    return dc_V_;
}

void Fftfilt::setWeight_Y(const std::string& value) {
    weight_Y_ = value;
}

std::string Fftfilt::getWeight_Y() const {
    return weight_Y_;
}

void Fftfilt::setWeight_U(const std::string& value) {
    weight_U_ = value;
}

std::string Fftfilt::getWeight_U() const {
    return weight_U_;
}

void Fftfilt::setWeight_V(const std::string& value) {
    weight_V_ = value;
}

std::string Fftfilt::getWeight_V() const {
    return weight_V_;
}

void Fftfilt::setEval(int value) {
    eval_ = value;
}

int Fftfilt::getEval() const {
    return eval_;
}

std::string Fftfilt::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "fftfilt";

    bool first = true;

    if (dc_Y_ != 0) {
        desc << (first ? "=" : ":") << "dc_Y=" << dc_Y_;
        first = false;
    }
    if (dc_U_ != 0) {
        desc << (first ? "=" : ":") << "dc_U=" << dc_U_;
        first = false;
    }
    if (dc_V_ != 0) {
        desc << (first ? "=" : ":") << "dc_V=" << dc_V_;
        first = false;
    }
    if (weight_Y_ != "1") {
        desc << (first ? "=" : ":") << "weight_Y=" << weight_Y_;
        first = false;
    }
    if (!weight_U_.empty()) {
        desc << (first ? "=" : ":") << "weight_U=" << weight_U_;
        first = false;
    }
    if (!weight_V_.empty()) {
        desc << (first ? "=" : ":") << "weight_V=" << weight_V_;
        first = false;
    }
    if (eval_ != 0) {
        desc << (first ? "=" : ":") << "eval=" << eval_;
        first = false;
    }

    return desc.str();
}
