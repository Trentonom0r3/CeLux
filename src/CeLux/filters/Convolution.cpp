#include "Convolution.hpp"
#include <sstream>

Convolution::Convolution(const std::string& _0m, const std::string& _1m, const std::string& _2m, const std::string& _3m, float _0rdiv, float _1rdiv, float _2rdiv, float _3rdiv, float _0bias, float _1bias, float _2bias, float _3bias, int _0mode, int _1mode, int _2mode, int _3mode) {
    // Initialize member variables from parameters
    this->_0m_ = _0m;
    this->_1m_ = _1m;
    this->_2m_ = _2m;
    this->_3m_ = _3m;
    this->_0rdiv_ = _0rdiv;
    this->_1rdiv_ = _1rdiv;
    this->_2rdiv_ = _2rdiv;
    this->_3rdiv_ = _3rdiv;
    this->_0bias_ = _0bias;
    this->_1bias_ = _1bias;
    this->_2bias_ = _2bias;
    this->_3bias_ = _3bias;
    this->_0mode_ = _0mode;
    this->_1mode_ = _1mode;
    this->_2mode_ = _2mode;
    this->_3mode_ = _3mode;
}

Convolution::~Convolution() {
    // Destructor implementation (if needed)
}

void Convolution::set_0m(const std::string& value) {
    _0m_ = value;
}

std::string Convolution::get_0m() const {
    return _0m_;
}

void Convolution::set_1m(const std::string& value) {
    _1m_ = value;
}

std::string Convolution::get_1m() const {
    return _1m_;
}

void Convolution::set_2m(const std::string& value) {
    _2m_ = value;
}

std::string Convolution::get_2m() const {
    return _2m_;
}

void Convolution::set_3m(const std::string& value) {
    _3m_ = value;
}

std::string Convolution::get_3m() const {
    return _3m_;
}

void Convolution::set_0rdiv(float value) {
    _0rdiv_ = value;
}

float Convolution::get_0rdiv() const {
    return _0rdiv_;
}

void Convolution::set_1rdiv(float value) {
    _1rdiv_ = value;
}

float Convolution::get_1rdiv() const {
    return _1rdiv_;
}

void Convolution::set_2rdiv(float value) {
    _2rdiv_ = value;
}

float Convolution::get_2rdiv() const {
    return _2rdiv_;
}

void Convolution::set_3rdiv(float value) {
    _3rdiv_ = value;
}

float Convolution::get_3rdiv() const {
    return _3rdiv_;
}

void Convolution::set_0bias(float value) {
    _0bias_ = value;
}

float Convolution::get_0bias() const {
    return _0bias_;
}

void Convolution::set_1bias(float value) {
    _1bias_ = value;
}

float Convolution::get_1bias() const {
    return _1bias_;
}

void Convolution::set_2bias(float value) {
    _2bias_ = value;
}

float Convolution::get_2bias() const {
    return _2bias_;
}

void Convolution::set_3bias(float value) {
    _3bias_ = value;
}

float Convolution::get_3bias() const {
    return _3bias_;
}

void Convolution::set_0mode(int value) {
    _0mode_ = value;
}

int Convolution::get_0mode() const {
    return _0mode_;
}

void Convolution::set_1mode(int value) {
    _1mode_ = value;
}

int Convolution::get_1mode() const {
    return _1mode_;
}

void Convolution::set_2mode(int value) {
    _2mode_ = value;
}

int Convolution::get_2mode() const {
    return _2mode_;
}

void Convolution::set_3mode(int value) {
    _3mode_ = value;
}

int Convolution::get_3mode() const {
    return _3mode_;
}

std::string Convolution::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "convolution";

    bool first = true;

    if (_0m_ != "0 0 0 0 1 0 0 0 0") {
        desc << (first ? "=" : ":") << "0m=" << _0m_;
        first = false;
    }
    if (_1m_ != "0 0 0 0 1 0 0 0 0") {
        desc << (first ? "=" : ":") << "1m=" << _1m_;
        first = false;
    }
    if (_2m_ != "0 0 0 0 1 0 0 0 0") {
        desc << (first ? "=" : ":") << "2m=" << _2m_;
        first = false;
    }
    if (_3m_ != "0 0 0 0 1 0 0 0 0") {
        desc << (first ? "=" : ":") << "3m=" << _3m_;
        first = false;
    }
    if (_0rdiv_ != 0.00) {
        desc << (first ? "=" : ":") << "0rdiv=" << _0rdiv_;
        first = false;
    }
    if (_1rdiv_ != 0.00) {
        desc << (first ? "=" : ":") << "1rdiv=" << _1rdiv_;
        first = false;
    }
    if (_2rdiv_ != 0.00) {
        desc << (first ? "=" : ":") << "2rdiv=" << _2rdiv_;
        first = false;
    }
    if (_3rdiv_ != 0.00) {
        desc << (first ? "=" : ":") << "3rdiv=" << _3rdiv_;
        first = false;
    }
    if (_0bias_ != 0.00) {
        desc << (first ? "=" : ":") << "0bias=" << _0bias_;
        first = false;
    }
    if (_1bias_ != 0.00) {
        desc << (first ? "=" : ":") << "1bias=" << _1bias_;
        first = false;
    }
    if (_2bias_ != 0.00) {
        desc << (first ? "=" : ":") << "2bias=" << _2bias_;
        first = false;
    }
    if (_3bias_ != 0.00) {
        desc << (first ? "=" : ":") << "3bias=" << _3bias_;
        first = false;
    }
    if (_0mode_ != 0) {
        desc << (first ? "=" : ":") << "0mode=" << _0mode_;
        first = false;
    }
    if (_1mode_ != 0) {
        desc << (first ? "=" : ":") << "1mode=" << _1mode_;
        first = false;
    }
    if (_2mode_ != 0) {
        desc << (first ? "=" : ":") << "2mode=" << _2mode_;
        first = false;
    }
    if (_3mode_ != 0) {
        desc << (first ? "=" : ":") << "3mode=" << _3mode_;
        first = false;
    }

    return desc.str();
}
