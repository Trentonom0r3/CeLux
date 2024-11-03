#include "Showcwt.hpp"
#include <sstream>

Showcwt::Showcwt(std::pair<int, int> size, const std::string& rate, int scale, int iscale, float min, float max, float imin, float imax, float logb, float deviation, int pps, int mode, int slide, int direction, float bar, float rotation) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->scale_ = scale;
    this->iscale_ = iscale;
    this->min_ = min;
    this->max_ = max;
    this->imin_ = imin;
    this->imax_ = imax;
    this->logb_ = logb;
    this->deviation_ = deviation;
    this->pps_ = pps;
    this->mode_ = mode;
    this->slide_ = slide;
    this->direction_ = direction;
    this->bar_ = bar;
    this->rotation_ = rotation;
}

Showcwt::~Showcwt() {
    // Destructor implementation (if needed)
}

void Showcwt::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Showcwt::getSize() const {
    return size_;
}

void Showcwt::setRate(const std::string& value) {
    rate_ = value;
}

std::string Showcwt::getRate() const {
    return rate_;
}

void Showcwt::setScale(int value) {
    scale_ = value;
}

int Showcwt::getScale() const {
    return scale_;
}

void Showcwt::setIscale(int value) {
    iscale_ = value;
}

int Showcwt::getIscale() const {
    return iscale_;
}

void Showcwt::setMin(float value) {
    min_ = value;
}

float Showcwt::getMin() const {
    return min_;
}

void Showcwt::setMax(float value) {
    max_ = value;
}

float Showcwt::getMax() const {
    return max_;
}

void Showcwt::setImin(float value) {
    imin_ = value;
}

float Showcwt::getImin() const {
    return imin_;
}

void Showcwt::setImax(float value) {
    imax_ = value;
}

float Showcwt::getImax() const {
    return imax_;
}

void Showcwt::setLogb(float value) {
    logb_ = value;
}

float Showcwt::getLogb() const {
    return logb_;
}

void Showcwt::setDeviation(float value) {
    deviation_ = value;
}

float Showcwt::getDeviation() const {
    return deviation_;
}

void Showcwt::setPps(int value) {
    pps_ = value;
}

int Showcwt::getPps() const {
    return pps_;
}

void Showcwt::setMode(int value) {
    mode_ = value;
}

int Showcwt::getMode() const {
    return mode_;
}

void Showcwt::setSlide(int value) {
    slide_ = value;
}

int Showcwt::getSlide() const {
    return slide_;
}

void Showcwt::setDirection(int value) {
    direction_ = value;
}

int Showcwt::getDirection() const {
    return direction_;
}

void Showcwt::setBar(float value) {
    bar_ = value;
}

float Showcwt::getBar() const {
    return bar_;
}

void Showcwt::setRotation(float value) {
    rotation_ = value;
}

float Showcwt::getRotation() const {
    return rotation_;
}

std::string Showcwt::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showcwt";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rate_ != "25") {
        desc << (first ? "=" : ":") << "rate=" << rate_;
        first = false;
    }
    if (scale_ != 0) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (iscale_ != 0) {
        desc << (first ? "=" : ":") << "iscale=" << iscale_;
        first = false;
    }
    if (min_ != 20.00) {
        desc << (first ? "=" : ":") << "min=" << min_;
        first = false;
    }
    if (max_ != 20000.00) {
        desc << (first ? "=" : ":") << "max=" << max_;
        first = false;
    }
    if (imin_ != 0.00) {
        desc << (first ? "=" : ":") << "imin=" << imin_;
        first = false;
    }
    if (imax_ != 1.00) {
        desc << (first ? "=" : ":") << "imax=" << imax_;
        first = false;
    }
    if (logb_ != 0.00) {
        desc << (first ? "=" : ":") << "logb=" << logb_;
        first = false;
    }
    if (deviation_ != 1.00) {
        desc << (first ? "=" : ":") << "deviation=" << deviation_;
        first = false;
    }
    if (pps_ != 64) {
        desc << (first ? "=" : ":") << "pps=" << pps_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (slide_ != 0) {
        desc << (first ? "=" : ":") << "slide=" << slide_;
        first = false;
    }
    if (direction_ != 0) {
        desc << (first ? "=" : ":") << "direction=" << direction_;
        first = false;
    }
    if (bar_ != 0.00) {
        desc << (first ? "=" : ":") << "bar=" << bar_;
        first = false;
    }
    if (rotation_ != 0.00) {
        desc << (first ? "=" : ":") << "rotation=" << rotation_;
        first = false;
    }

    return desc.str();
}
