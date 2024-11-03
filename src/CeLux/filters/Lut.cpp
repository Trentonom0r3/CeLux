#include "Lut.hpp"
#include <sstream>

Lut::Lut(const std::string& c0, const std::string& c1, const std::string& c2, const std::string& c3, const std::string& y, const std::string& u, const std::string& v, const std::string& r, const std::string& g, const std::string& b, const std::string& a) {
    // Initialize member variables from parameters
    this->c0_ = c0;
    this->c1_ = c1;
    this->c2_ = c2;
    this->c3_ = c3;
    this->y_ = y;
    this->u_ = u;
    this->v_ = v;
    this->r_ = r;
    this->g_ = g;
    this->b_ = b;
    this->a_ = a;
}

Lut::~Lut() {
    // Destructor implementation (if needed)
}

void Lut::setC0(const std::string& value) {
    c0_ = value;
}

std::string Lut::getC0() const {
    return c0_;
}

void Lut::setC1(const std::string& value) {
    c1_ = value;
}

std::string Lut::getC1() const {
    return c1_;
}

void Lut::setC2(const std::string& value) {
    c2_ = value;
}

std::string Lut::getC2() const {
    return c2_;
}

void Lut::setC3(const std::string& value) {
    c3_ = value;
}

std::string Lut::getC3() const {
    return c3_;
}

void Lut::setY(const std::string& value) {
    y_ = value;
}

std::string Lut::getY() const {
    return y_;
}

void Lut::setU(const std::string& value) {
    u_ = value;
}

std::string Lut::getU() const {
    return u_;
}

void Lut::setV(const std::string& value) {
    v_ = value;
}

std::string Lut::getV() const {
    return v_;
}

void Lut::setR(const std::string& value) {
    r_ = value;
}

std::string Lut::getR() const {
    return r_;
}

void Lut::setG(const std::string& value) {
    g_ = value;
}

std::string Lut::getG() const {
    return g_;
}

void Lut::setB(const std::string& value) {
    b_ = value;
}

std::string Lut::getB() const {
    return b_;
}

void Lut::setA(const std::string& value) {
    a_ = value;
}

std::string Lut::getA() const {
    return a_;
}

std::string Lut::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "lut";

    bool first = true;

    if (c0_ != "clipval") {
        desc << (first ? "=" : ":") << "c0=" << c0_;
        first = false;
    }
    if (c1_ != "clipval") {
        desc << (first ? "=" : ":") << "c1=" << c1_;
        first = false;
    }
    if (c2_ != "clipval") {
        desc << (first ? "=" : ":") << "c2=" << c2_;
        first = false;
    }
    if (c3_ != "clipval") {
        desc << (first ? "=" : ":") << "c3=" << c3_;
        first = false;
    }
    if (y_ != "clipval") {
        desc << (first ? "=" : ":") << "y=" << y_;
        first = false;
    }
    if (u_ != "clipval") {
        desc << (first ? "=" : ":") << "u=" << u_;
        first = false;
    }
    if (v_ != "clipval") {
        desc << (first ? "=" : ":") << "v=" << v_;
        first = false;
    }
    if (r_ != "clipval") {
        desc << (first ? "=" : ":") << "r=" << r_;
        first = false;
    }
    if (g_ != "clipval") {
        desc << (first ? "=" : ":") << "g=" << g_;
        first = false;
    }
    if (b_ != "clipval") {
        desc << (first ? "=" : ":") << "b=" << b_;
        first = false;
    }
    if (a_ != "clipval") {
        desc << (first ? "=" : ":") << "a=" << a_;
        first = false;
    }

    return desc.str();
}
