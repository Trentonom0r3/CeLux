#include "Gradients.hpp"
#include <sstream>

Gradients::Gradients(std::pair<int, int> size, std::pair<int, int> rate, const std::string& c0, const std::string& c1, const std::string& c2, const std::string& c3, const std::string& c4, const std::string& c5, const std::string& c6, const std::string& c7, int x0, int y0, int x1, int y1, int nb_colors, int64_t seed, int64_t duration, float speed, int type) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->c0_ = c0;
    this->c1_ = c1;
    this->c2_ = c2;
    this->c3_ = c3;
    this->c4_ = c4;
    this->c5_ = c5;
    this->c6_ = c6;
    this->c7_ = c7;
    this->x0_ = x0;
    this->y0_ = y0;
    this->x1_ = x1;
    this->y1_ = y1;
    this->nb_colors_ = nb_colors;
    this->seed_ = seed;
    this->duration_ = duration;
    this->speed_ = speed;
    this->type_ = type;
}

Gradients::~Gradients() {
    // Destructor implementation (if needed)
}

void Gradients::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Gradients::getSize() const {
    return size_;
}

void Gradients::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Gradients::getRate() const {
    return rate_;
}

void Gradients::setC0(const std::string& value) {
    c0_ = value;
}

std::string Gradients::getC0() const {
    return c0_;
}

void Gradients::setC1(const std::string& value) {
    c1_ = value;
}

std::string Gradients::getC1() const {
    return c1_;
}

void Gradients::setC2(const std::string& value) {
    c2_ = value;
}

std::string Gradients::getC2() const {
    return c2_;
}

void Gradients::setC3(const std::string& value) {
    c3_ = value;
}

std::string Gradients::getC3() const {
    return c3_;
}

void Gradients::setC4(const std::string& value) {
    c4_ = value;
}

std::string Gradients::getC4() const {
    return c4_;
}

void Gradients::setC5(const std::string& value) {
    c5_ = value;
}

std::string Gradients::getC5() const {
    return c5_;
}

void Gradients::setC6(const std::string& value) {
    c6_ = value;
}

std::string Gradients::getC6() const {
    return c6_;
}

void Gradients::setC7(const std::string& value) {
    c7_ = value;
}

std::string Gradients::getC7() const {
    return c7_;
}

void Gradients::setX0(int value) {
    x0_ = value;
}

int Gradients::getX0() const {
    return x0_;
}

void Gradients::setY0(int value) {
    y0_ = value;
}

int Gradients::getY0() const {
    return y0_;
}

void Gradients::setX1(int value) {
    x1_ = value;
}

int Gradients::getX1() const {
    return x1_;
}

void Gradients::setY1(int value) {
    y1_ = value;
}

int Gradients::getY1() const {
    return y1_;
}

void Gradients::setNb_colors(int value) {
    nb_colors_ = value;
}

int Gradients::getNb_colors() const {
    return nb_colors_;
}

void Gradients::setSeed(int64_t value) {
    seed_ = value;
}

int64_t Gradients::getSeed() const {
    return seed_;
}

void Gradients::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Gradients::getDuration() const {
    return duration_;
}

void Gradients::setSpeed(float value) {
    speed_ = value;
}

float Gradients::getSpeed() const {
    return speed_;
}

void Gradients::setType(int value) {
    type_ = value;
}

int Gradients::getType() const {
    return type_;
}

std::string Gradients::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "gradients";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (c0_ != "random") {
        desc << (first ? "=" : ":") << "c0=" << c0_;
        first = false;
    }
    if (c1_ != "random") {
        desc << (first ? "=" : ":") << "c1=" << c1_;
        first = false;
    }
    if (c2_ != "random") {
        desc << (first ? "=" : ":") << "c2=" << c2_;
        first = false;
    }
    if (c3_ != "random") {
        desc << (first ? "=" : ":") << "c3=" << c3_;
        first = false;
    }
    if (c4_ != "random") {
        desc << (first ? "=" : ":") << "c4=" << c4_;
        first = false;
    }
    if (c5_ != "random") {
        desc << (first ? "=" : ":") << "c5=" << c5_;
        first = false;
    }
    if (c6_ != "random") {
        desc << (first ? "=" : ":") << "c6=" << c6_;
        first = false;
    }
    if (c7_ != "random") {
        desc << (first ? "=" : ":") << "c7=" << c7_;
        first = false;
    }
    if (x0_ != -1) {
        desc << (first ? "=" : ":") << "x0=" << x0_;
        first = false;
    }
    if (y0_ != -1) {
        desc << (first ? "=" : ":") << "y0=" << y0_;
        first = false;
    }
    if (x1_ != -1) {
        desc << (first ? "=" : ":") << "x1=" << x1_;
        first = false;
    }
    if (y1_ != -1) {
        desc << (first ? "=" : ":") << "y1=" << y1_;
        first = false;
    }
    if (nb_colors_ != 2) {
        desc << (first ? "=" : ":") << "nb_colors=" << nb_colors_;
        first = false;
    }
    if (seed_ != 0) {
        desc << (first ? "=" : ":") << "seed=" << seed_;
        first = false;
    }
    if (duration_ != 0) {
        desc << (first ? "=" : ":") << "duration=" << duration_;
        first = false;
    }
    if (speed_ != 0.01) {
        desc << (first ? "=" : ":") << "speed=" << speed_;
        first = false;
    }
    if (type_ != 0) {
        desc << (first ? "=" : ":") << "type=" << type_;
        first = false;
    }

    return desc.str();
}
