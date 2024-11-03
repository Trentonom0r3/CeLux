#include "Thistogram.hpp"
#include <sstream>

Thistogram::Thistogram(int width, int display_mode, int levels_mode, int components, float bgopacity, bool envelope, const std::string& ecolor, int slide) {
    // Initialize member variables from parameters
    this->width_ = width;
    this->display_mode_ = display_mode;
    this->levels_mode_ = levels_mode;
    this->components_ = components;
    this->bgopacity_ = bgopacity;
    this->envelope_ = envelope;
    this->ecolor_ = ecolor;
    this->slide_ = slide;
}

Thistogram::~Thistogram() {
    // Destructor implementation (if needed)
}

void Thistogram::setWidth(int value) {
    width_ = value;
}

int Thistogram::getWidth() const {
    return width_;
}

void Thistogram::setDisplay_mode(int value) {
    display_mode_ = value;
}

int Thistogram::getDisplay_mode() const {
    return display_mode_;
}

void Thistogram::setLevels_mode(int value) {
    levels_mode_ = value;
}

int Thistogram::getLevels_mode() const {
    return levels_mode_;
}

void Thistogram::setComponents(int value) {
    components_ = value;
}

int Thistogram::getComponents() const {
    return components_;
}

void Thistogram::setBgopacity(float value) {
    bgopacity_ = value;
}

float Thistogram::getBgopacity() const {
    return bgopacity_;
}

void Thistogram::setEnvelope(bool value) {
    envelope_ = value;
}

bool Thistogram::getEnvelope() const {
    return envelope_;
}

void Thistogram::setEcolor(const std::string& value) {
    ecolor_ = value;
}

std::string Thistogram::getEcolor() const {
    return ecolor_;
}

void Thistogram::setSlide(int value) {
    slide_ = value;
}

int Thistogram::getSlide() const {
    return slide_;
}

std::string Thistogram::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "thistogram";

    bool first = true;

    if (width_ != 0) {
        desc << (first ? "=" : ":") << "width=" << width_;
        first = false;
    }
    if (display_mode_ != 2) {
        desc << (first ? "=" : ":") << "display_mode=" << display_mode_;
        first = false;
    }
    if (levels_mode_ != 0) {
        desc << (first ? "=" : ":") << "levels_mode=" << levels_mode_;
        first = false;
    }
    if (components_ != 7) {
        desc << (first ? "=" : ":") << "components=" << components_;
        first = false;
    }
    if (bgopacity_ != 0.90) {
        desc << (first ? "=" : ":") << "bgopacity=" << bgopacity_;
        first = false;
    }
    if (envelope_ != false) {
        desc << (first ? "=" : ":") << "envelope=" << (envelope_ ? "1" : "0");
        first = false;
    }
    if (ecolor_ != "gold") {
        desc << (first ? "=" : ":") << "ecolor=" << ecolor_;
        first = false;
    }
    if (slide_ != 1) {
        desc << (first ? "=" : ":") << "slide=" << slide_;
        first = false;
    }

    return desc.str();
}
