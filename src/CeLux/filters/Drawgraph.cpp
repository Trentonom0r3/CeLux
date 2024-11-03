#include "Drawgraph.hpp"
#include <sstream>

Drawgraph::Drawgraph(const std::string& m1, const std::string& fg1, const std::string& m2, const std::string& fg2, const std::string& m3, const std::string& fg3, const std::string& m4, const std::string& fg4, const std::string& bg, float min, float max, int mode, int slide, std::pair<int, int> size, std::pair<int, int> rate) {
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

Drawgraph::~Drawgraph() {
    // Destructor implementation (if needed)
}

void Drawgraph::setM1(const std::string& value) {
    m1_ = value;
}

std::string Drawgraph::getM1() const {
    return m1_;
}

void Drawgraph::setFg1(const std::string& value) {
    fg1_ = value;
}

std::string Drawgraph::getFg1() const {
    return fg1_;
}

void Drawgraph::setM2(const std::string& value) {
    m2_ = value;
}

std::string Drawgraph::getM2() const {
    return m2_;
}

void Drawgraph::setFg2(const std::string& value) {
    fg2_ = value;
}

std::string Drawgraph::getFg2() const {
    return fg2_;
}

void Drawgraph::setM3(const std::string& value) {
    m3_ = value;
}

std::string Drawgraph::getM3() const {
    return m3_;
}

void Drawgraph::setFg3(const std::string& value) {
    fg3_ = value;
}

std::string Drawgraph::getFg3() const {
    return fg3_;
}

void Drawgraph::setM4(const std::string& value) {
    m4_ = value;
}

std::string Drawgraph::getM4() const {
    return m4_;
}

void Drawgraph::setFg4(const std::string& value) {
    fg4_ = value;
}

std::string Drawgraph::getFg4() const {
    return fg4_;
}

void Drawgraph::setBg(const std::string& value) {
    bg_ = value;
}

std::string Drawgraph::getBg() const {
    return bg_;
}

void Drawgraph::setMin(float value) {
    min_ = value;
}

float Drawgraph::getMin() const {
    return min_;
}

void Drawgraph::setMax(float value) {
    max_ = value;
}

float Drawgraph::getMax() const {
    return max_;
}

void Drawgraph::setMode(int value) {
    mode_ = value;
}

int Drawgraph::getMode() const {
    return mode_;
}

void Drawgraph::setSlide(int value) {
    slide_ = value;
}

int Drawgraph::getSlide() const {
    return slide_;
}

void Drawgraph::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Drawgraph::getSize() const {
    return size_;
}

void Drawgraph::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Drawgraph::getRate() const {
    return rate_;
}

std::string Drawgraph::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "drawgraph";

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
