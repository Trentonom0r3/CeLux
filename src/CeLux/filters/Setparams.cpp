#include "Setparams.hpp"
#include <sstream>

Setparams::Setparams(int field_mode, int range, int color_primaries, int color_trc, int colorspace) {
    // Initialize member variables from parameters
    this->field_mode_ = field_mode;
    this->range_ = range;
    this->color_primaries_ = color_primaries;
    this->color_trc_ = color_trc;
    this->colorspace_ = colorspace;
}

Setparams::~Setparams() {
    // Destructor implementation (if needed)
}

void Setparams::setField_mode(int value) {
    field_mode_ = value;
}

int Setparams::getField_mode() const {
    return field_mode_;
}

void Setparams::setRange(int value) {
    range_ = value;
}

int Setparams::getRange() const {
    return range_;
}

void Setparams::setColor_primaries(int value) {
    color_primaries_ = value;
}

int Setparams::getColor_primaries() const {
    return color_primaries_;
}

void Setparams::setColor_trc(int value) {
    color_trc_ = value;
}

int Setparams::getColor_trc() const {
    return color_trc_;
}

void Setparams::setColorspace(int value) {
    colorspace_ = value;
}

int Setparams::getColorspace() const {
    return colorspace_;
}

std::string Setparams::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "setparams";

    bool first = true;

    if (field_mode_ != -1) {
        desc << (first ? "=" : ":") << "field_mode=" << field_mode_;
        first = false;
    }
    if (range_ != -1) {
        desc << (first ? "=" : ":") << "range=" << range_;
        first = false;
    }
    if (color_primaries_ != -1) {
        desc << (first ? "=" : ":") << "color_primaries=" << color_primaries_;
        first = false;
    }
    if (color_trc_ != -1) {
        desc << (first ? "=" : ":") << "color_trc=" << color_trc_;
        first = false;
    }
    if (colorspace_ != -1) {
        desc << (first ? "=" : ":") << "colorspace=" << colorspace_;
        first = false;
    }

    return desc.str();
}
