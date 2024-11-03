#include "Adrawgraph.hpp"
#include <sstream>

Adrawgraph::Adrawgraph(const std::string& m1, const std::string& fg1, const std::string& m2, const std::string& fg2, const std::string& m3, const std::string& fg3, const std::string& m4, const std::string& fg4, const std::string& bg, float min, float max, int mode, int slide, std::pair<int, int> size, std::pair<int, int> rate) {
    // Initialize member variables from parameters
    this->m1_ = m1;
    this->fg1_ = fg1;
    this->m2_ = m2;
    this->fg2_ = fg2;
    this->m3_ = m3;
    this->fg3_ = fg3;
    this->m4_ = m4;
    this->fg4_ = fg4;
    this->bg_ = bg;
    this->min_ = min;
    this->max_ = max;
    this->mode_ = mode;
    this->slide_ = slide;
    this->size_ = size;
    this->rate_ = rate;
}

Adrawgraph::~Adrawgraph() {
    // Destructor implementation (if needed)
}

void Adrawgraph::setM1(const std::string& value) {
    m1_ = value;
}

std::string Adrawgraph::getM1() const {
    return m1_;
}

void Adrawgraph::setFg1(const std::string& value) {
    fg1_ = value;
}

std::string Adrawgraph::getFg1() const {
    return fg1_;
}

void Adrawgraph::setM2(const std::string& value) {
    m2_ = value;
}

std::string Adrawgraph::getM2() const {
    return m2_;
}

void Adrawgraph::setFg2(const std::string& value) {
    fg2_ = value;
}

std::string Adrawgraph::getFg2() const {
    return fg2_;
}

void Adrawgraph::setM3(const std::string& value) {
    m3_ = value;
}

std::string Adrawgraph::getM3() const {
    return m3_;
}

void Adrawgraph::setFg3(const std::string& value) {
    fg3_ = value;
}

std::string Adrawgraph::getFg3() const {
    return fg3_;
}

void Adrawgraph::setM4(const std::string& value) {
    m4_ = value;
}

std::string Adrawgraph::getM4() const {
    return m4_;
}

void Adrawgraph::setFg4(const std::string& value) {
    fg4_ = value;
}

std::string Adrawgraph::getFg4() const {
    return fg4_;
}

void Adrawgraph::setBg(const std::string& value) {
    bg_ = value;
}

std::string Adrawgraph::getBg() const {
    return bg_;
}

void Adrawgraph::setMin(float value) {
    min_ = value;
}

float Adrawgraph::getMin() const {
    return min_;
}

void Adrawgraph::setMax(float value) {
    max_ = value;
}

float Adrawgraph::getMax() const {
    return max_;
}

void Adrawgraph::setMode(int value) {
    mode_ = value;
}

int Adrawgraph::getMode() const {
    return mode_;
}

void Adrawgraph::setSlide(int value) {
    slide_ = value;
}

int Adrawgraph::getSlide() const {
    return slide_;
}

void Adrawgraph::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Adrawgraph::getSize() const {
    return size_;
}

void Adrawgraph::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Adrawgraph::getRate() const {
    return rate_;
}

std::string Adrawgraph::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "adrawgraph";

    bool first = true;

    if (!m1_.empty()) {
        desc << (first ? "=" : ":") << "m1=" << m1_;
        first = false;
    }
    if (fg1_ != "0xffff0000") {
        desc << (first ? "=" : ":") << "fg1=" << fg1_;
        first = false;
    }
    if (!m2_.empty()) {
        desc << (first ? "=" : ":") << "m2=" << m2_;
        first = false;
    }
    if (fg2_ != "0xff00ff00") {
        desc << (first ? "=" : ":") << "fg2=" << fg2_;
        first = false;
    }
    if (!m3_.empty()) {
        desc << (first ? "=" : ":") << "m3=" << m3_;
        first = false;
    }
    if (fg3_ != "0xffff00ff") {
        desc << (first ? "=" : ":") << "fg3=" << fg3_;
        first = false;
    }
    if (!m4_.empty()) {
        desc << (first ? "=" : ":") << "m4=" << m4_;
        first = false;
    }
    if (fg4_ != "0xffffff00") {
        desc << (first ? "=" : ":") << "fg4=" << fg4_;
        first = false;
    }
    if (bg_ != "white") {
        desc << (first ? "=" : ":") << "bg=" << bg_;
        first = false;
    }
    if (min_ != -1.00) {
        desc << (first ? "=" : ":") << "min=" << min_;
        first = false;
    }
    if (max_ != 1.00) {
        desc << (first ? "=" : ":") << "max=" << max_;
        first = false;
    }
    if (mode_ != 2) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (slide_ != 0) {
        desc << (first ? "=" : ":") << "slide=" << slide_;
        first = false;
    }
    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }

    return desc.str();
}
