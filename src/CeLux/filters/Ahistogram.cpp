#include "Ahistogram.hpp"
#include <sstream>

Ahistogram::Ahistogram(int dmode, std::pair<int, int> rate, std::pair<int, int> size, int scale, int ascale, int acount, float rheight, int slide, int hmode) {
    // Initialize member variables from parameters
    this->dmode_ = dmode;
    this->rate_ = rate;
    this->size_ = size;
    this->scale_ = scale;
    this->ascale_ = ascale;
    this->acount_ = acount;
    this->rheight_ = rheight;
    this->slide_ = slide;
    this->hmode_ = hmode;
}

Ahistogram::~Ahistogram() {
    // Destructor implementation (if needed)
}

void Ahistogram::setDmode(int value) {
    dmode_ = value;
}

int Ahistogram::getDmode() const {
    return dmode_;
}

void Ahistogram::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Ahistogram::getRate() const {
    return rate_;
}

void Ahistogram::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Ahistogram::getSize() const {
    return size_;
}

void Ahistogram::setScale(int value) {
    scale_ = value;
}

int Ahistogram::getScale() const {
    return scale_;
}

void Ahistogram::setAscale(int value) {
    ascale_ = value;
}

int Ahistogram::getAscale() const {
    return ascale_;
}

void Ahistogram::setAcount(int value) {
    acount_ = value;
}

int Ahistogram::getAcount() const {
    return acount_;
}

void Ahistogram::setRheight(float value) {
    rheight_ = value;
}

float Ahistogram::getRheight() const {
    return rheight_;
}

void Ahistogram::setSlide(int value) {
    slide_ = value;
}

int Ahistogram::getSlide() const {
    return slide_;
}

void Ahistogram::setHmode(int value) {
    hmode_ = value;
}

int Ahistogram::getHmode() const {
    return hmode_;
}

std::string Ahistogram::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "ahistogram";

    bool first = true;

    if (dmode_ != 0) {
        desc << (first ? "=" : ":") << "dmode=" << dmode_;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (scale_ != 3) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (ascale_ != 1) {
        desc << (first ? "=" : ":") << "ascale=" << ascale_;
        first = false;
    }
    if (acount_ != 1) {
        desc << (first ? "=" : ":") << "acount=" << acount_;
        first = false;
    }
    if (rheight_ != 0.10) {
        desc << (first ? "=" : ":") << "rheight=" << rheight_;
        first = false;
    }
    if (slide_ != 0) {
        desc << (first ? "=" : ":") << "slide=" << slide_;
        first = false;
    }
    if (hmode_ != 0) {
        desc << (first ? "=" : ":") << "hmode=" << hmode_;
        first = false;
    }

    return desc.str();
}
