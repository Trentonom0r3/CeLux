#include "Histogram.hpp"
#include <sstream>

Histogram::Histogram(int level_height, int scale_height, int display_mode, int levels_mode, int components, float fgopacity, float bgopacity, int colors_mode) {
    // Initialize member variables from parameters
    this->level_height_ = level_height;
    this->scale_height_ = scale_height;
    this->display_mode_ = display_mode;
    this->levels_mode_ = levels_mode;
    this->components_ = components;
    this->fgopacity_ = fgopacity;
    this->bgopacity_ = bgopacity;
    this->colors_mode_ = colors_mode;
}

Histogram::~Histogram() {
    // Destructor implementation (if needed)
}

void Histogram::setLevel_height(int value) {
    level_height_ = value;
}

int Histogram::getLevel_height() const {
    return level_height_;
}

void Histogram::setScale_height(int value) {
    scale_height_ = value;
}

int Histogram::getScale_height() const {
    return scale_height_;
}

void Histogram::setDisplay_mode(int value) {
    display_mode_ = value;
}

int Histogram::getDisplay_mode() const {
    return display_mode_;
}

void Histogram::setLevels_mode(int value) {
    levels_mode_ = value;
}

int Histogram::getLevels_mode() const {
    return levels_mode_;
}

void Histogram::setComponents(int value) {
    components_ = value;
}

int Histogram::getComponents() const {
    return components_;
}

void Histogram::setFgopacity(float value) {
    fgopacity_ = value;
}

float Histogram::getFgopacity() const {
    return fgopacity_;
}

void Histogram::setBgopacity(float value) {
    bgopacity_ = value;
}

float Histogram::getBgopacity() const {
    return bgopacity_;
}

void Histogram::setColors_mode(int value) {
    colors_mode_ = value;
}

int Histogram::getColors_mode() const {
    return colors_mode_;
}

std::string Histogram::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "histogram";

    bool first = true;

    if (level_height_ != 200) {
        desc << (first ? "=" : ":") << "level_height=" << level_height_;
        first = false;
    }
    if (scale_height_ != 12) {
        desc << (first ? "=" : ":") << "scale_height=" << scale_height_;
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
    if (fgopacity_ != 0.70) {
        desc << (first ? "=" : ":") << "fgopacity=" << fgopacity_;
        first = false;
    }
    if (bgopacity_ != 0.50) {
        desc << (first ? "=" : ":") << "bgopacity=" << bgopacity_;
        first = false;
    }
    if (colors_mode_ != 0) {
        desc << (first ? "=" : ":") << "colors_mode=" << colors_mode_;
        first = false;
    }

    return desc.str();
}
