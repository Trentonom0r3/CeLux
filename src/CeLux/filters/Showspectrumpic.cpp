#include "Showspectrumpic.hpp"
#include <sstream>

Showspectrumpic::Showspectrumpic(std::pair<int, int> size, int mode, int color, int scale, int fscale, float saturation, int win_func, int orientation, float gain, bool legend, float rotation, int start, int stop, float drange, float limit, float opacity) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->mode_ = mode;
    this->color_ = color;
    this->scale_ = scale;
    this->fscale_ = fscale;
    this->saturation_ = saturation;
    this->win_func_ = win_func;
    this->orientation_ = orientation;
    this->gain_ = gain;
    this->legend_ = legend;
    this->rotation_ = rotation;
    this->start_ = start;
    this->stop_ = stop;
    this->drange_ = drange;
    this->limit_ = limit;
    this->opacity_ = opacity;
}

Showspectrumpic::~Showspectrumpic() {
    // Destructor implementation (if needed)
}

void Showspectrumpic::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Showspectrumpic::getSize() const {
    return size_;
}

void Showspectrumpic::setMode(int value) {
    mode_ = value;
}

int Showspectrumpic::getMode() const {
    return mode_;
}

void Showspectrumpic::setColor(int value) {
    color_ = value;
}

int Showspectrumpic::getColor() const {
    return color_;
}

void Showspectrumpic::setScale(int value) {
    scale_ = value;
}

int Showspectrumpic::getScale() const {
    return scale_;
}

void Showspectrumpic::setFscale(int value) {
    fscale_ = value;
}

int Showspectrumpic::getFscale() const {
    return fscale_;
}

void Showspectrumpic::setSaturation(float value) {
    saturation_ = value;
}

float Showspectrumpic::getSaturation() const {
    return saturation_;
}

void Showspectrumpic::setWin_func(int value) {
    win_func_ = value;
}

int Showspectrumpic::getWin_func() const {
    return win_func_;
}

void Showspectrumpic::setOrientation(int value) {
    orientation_ = value;
}

int Showspectrumpic::getOrientation() const {
    return orientation_;
}

void Showspectrumpic::setGain(float value) {
    gain_ = value;
}

float Showspectrumpic::getGain() const {
    return gain_;
}

void Showspectrumpic::setLegend(bool value) {
    legend_ = value;
}

bool Showspectrumpic::getLegend() const {
    return legend_;
}

void Showspectrumpic::setRotation(float value) {
    rotation_ = value;
}

float Showspectrumpic::getRotation() const {
    return rotation_;
}

void Showspectrumpic::setStart(int value) {
    start_ = value;
}

int Showspectrumpic::getStart() const {
    return start_;
}

void Showspectrumpic::setStop(int value) {
    stop_ = value;
}

int Showspectrumpic::getStop() const {
    return stop_;
}

void Showspectrumpic::setDrange(float value) {
    drange_ = value;
}

float Showspectrumpic::getDrange() const {
    return drange_;
}

void Showspectrumpic::setLimit(float value) {
    limit_ = value;
}

float Showspectrumpic::getLimit() const {
    return limit_;
}

void Showspectrumpic::setOpacity(float value) {
    opacity_ = value;
}

float Showspectrumpic::getOpacity() const {
    return opacity_;
}

std::string Showspectrumpic::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showspectrumpic";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (color_ != 1) {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }
    if (scale_ != 3) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (fscale_ != 0) {
        desc << (first ? "=" : ":") << "fscale=" << fscale_;
        first = false;
    }
    if (saturation_ != 1.00) {
        desc << (first ? "=" : ":") << "saturation=" << saturation_;
        first = false;
    }
    if (win_func_ != 1) {
        desc << (first ? "=" : ":") << "win_func=" << win_func_;
        first = false;
    }
    if (orientation_ != 0) {
        desc << (first ? "=" : ":") << "orientation=" << orientation_;
        first = false;
    }
    if (gain_ != 1.00) {
        desc << (first ? "=" : ":") << "gain=" << gain_;
        first = false;
    }
    if (legend_ != true) {
        desc << (first ? "=" : ":") << "legend=" << (legend_ ? "1" : "0");
        first = false;
    }
    if (rotation_ != 0.00) {
        desc << (first ? "=" : ":") << "rotation=" << rotation_;
        first = false;
    }
    if (start_ != 0) {
        desc << (first ? "=" : ":") << "start=" << start_;
        first = false;
    }
    if (stop_ != 0) {
        desc << (first ? "=" : ":") << "stop=" << stop_;
        first = false;
    }
    if (drange_ != 120.00) {
        desc << (first ? "=" : ":") << "drange=" << drange_;
        first = false;
    }
    if (limit_ != 0.00) {
        desc << (first ? "=" : ":") << "limit=" << limit_;
        first = false;
    }
    if (opacity_ != 1.00) {
        desc << (first ? "=" : ":") << "opacity=" << opacity_;
        first = false;
    }

    return desc.str();
}
