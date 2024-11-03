#include "Lutrgb.hpp"
#include <sstream>

Lutrgb::Lutrgb(const std::string& c0, const std::string& c1, const std::string& c2, const std::string& c3, const std::string& y, const std::string& u, const std::string& v, const std::string& r, const std::string& g, const std::string& b, const std::string& a) {
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

Lutrgb::~Lutrgb() {
    // Destructor implementation (if needed)
}

void Lutrgb::setC0(const std::string& value) {
    c0_ = value;
}

std::string Lutrgb::getC0() const {
    return c0_;
}

void Lutrgb::setC1(const std::string& value) {
    c1_ = value;
}

std::string Lutrgb::getC1() const {
    return c1_;
}

void Lutrgb::setC2(const std::string& value) {
    c2_ = value;
}

std::string Lutrgb::getC2() const {
    return c2_;
}

void Lutrgb::setC3(const std::string& value) {
    c3_ = value;
}

std::string Lutrgb::getC3() const {
    return c3_;
}

void Lutrgb::setY(const std::string& value) {
    y_ = value;
}

std::string Lutrgb::getY() const {
    return y_;
}

void Lutrgb::setU(const std::string& value) {
    u_ = value;
}

std::string Lutrgb::getU() const {
    return u_;
}

void Lutrgb::setV(const std::string& value) {
    v_ = value;
}

std::string Lutrgb::getV() const {
    return v_;
}

void Lutrgb::setR(const std::string& value) {
    r_ = value;
}

std::string Lutrgb::getR() const {
    return r_;
}

void Lutrgb::setG(const std::string& value) {
    g_ = value;
}

std::string Lutrgb::getG() const {
    return g_;
}

void Lutrgb::setB(const std::string& value) {
    b_ = value;
}

std::string Lutrgb::getB() const {
    return b_;
}

void Lutrgb::setA(const std::string& value) {
    a_ = value;
}

std::string Lutrgb::getA() const {
    return a_;
}

std::string Lutrgb::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "lutrgb";

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
