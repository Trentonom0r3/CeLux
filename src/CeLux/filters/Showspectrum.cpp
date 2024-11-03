#include "Showspectrum.hpp"
#include <sstream>

Showspectrum::Showspectrum(std::pair<int, int> size, int slide, int mode, int color, int scale, int fscale, float saturation, int win_func, int orientation, float overlap, float gain, int data, float rotation, int start, int stop, const std::string& fps, bool legend, float drange, float limit, float opacity) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->slide_ = slide;
    this->mode_ = mode;
    this->color_ = color;
    this->scale_ = scale;
    this->fscale_ = fscale;
    this->saturation_ = saturation;
    this->win_func_ = win_func;
    this->orientation_ = orientation;
    this->overlap_ = overlap;
    this->gain_ = gain;
    this->data_ = data;
    this->rotation_ = rotation;
    this->start_ = start;
    this->stop_ = stop;
    this->fps_ = fps;
    this->legend_ = legend;
    this->drange_ = drange;
    this->limit_ = limit;
    this->opacity_ = opacity;
}

Showspectrum::~Showspectrum() {
    // Destructor implementation (if needed)
}

void Showspectrum::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Showspectrum::getSize() const {
    return size_;
}

void Showspectrum::setSlide(int value) {
    slide_ = value;
}

int Showspectrum::getSlide() const {
    return slide_;
}

void Showspectrum::setMode(int value) {
    mode_ = value;
}

int Showspectrum::getMode() const {
    return mode_;
}

void Showspectrum::setColor(int value) {
    color_ = value;
}

int Showspectrum::getColor() const {
    return color_;
}

void Showspectrum::setScale(int value) {
    scale_ = value;
}

int Showspectrum::getScale() const {
    return scale_;
}

void Showspectrum::setFscale(int value) {
    fscale_ = value;
}

int Showspectrum::getFscale() const {
    return fscale_;
}

void Showspectrum::setSaturation(float value) {
    saturation_ = value;
}

float Showspectrum::getSaturation() const {
    return saturation_;
}

void Showspectrum::setWin_func(int value) {
    win_func_ = value;
}

int Showspectrum::getWin_func() const {
    return win_func_;
}

void Showspectrum::setOrientation(int value) {
    orientation_ = value;
}

int Showspectrum::getOrientation() const {
    return orientation_;
}

void Showspectrum::setOverlap(float value) {
    overlap_ = value;
}

float Showspectrum::getOverlap() const {
    return overlap_;
}

void Showspectrum::setGain(float value) {
    gain_ = value;
}

float Showspectrum::getGain() const {
    return gain_;
}

void Showspectrum::setData(int value) {
    data_ = value;
}

int Showspectrum::getData() const {
    return data_;
}

void Showspectrum::setRotation(float value) {
    rotation_ = value;
}

float Showspectrum::getRotation() const {
    return rotation_;
}

void Showspectrum::setStart(int value) {
    start_ = value;
}

int Showspectrum::getStart() const {
    return start_;
}

void Showspectrum::setStop(int value) {
    stop_ = value;
}

int Showspectrum::getStop() const {
    return stop_;
}

void Showspectrum::setFps(const std::string& value) {
    fps_ = value;
}

std::string Showspectrum::getFps() const {
    return fps_;
}

void Showspectrum::setLegend(bool value) {
    legend_ = value;
}

bool Showspectrum::getLegend() const {
    return legend_;
}

void Showspectrum::setDrange(float value) {
    drange_ = value;
}

float Showspectrum::getDrange() const {
    return drange_;
}

void Showspectrum::setLimit(float value) {
    limit_ = value;
}

float Showspectrum::getLimit() const {
    return limit_;
}

void Showspectrum::setOpacity(float value) {
    opacity_ = value;
}

float Showspectrum::getOpacity() const {
    return opacity_;
}

std::string Showspectrum::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showspectrum";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (slide_ != 0) {
        desc << (first ? "=" : ":") << "slide=" << slide_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (color_ != 0) {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }
    if (scale_ != 1) {
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
    if (overlap_ != 0.00) {
        desc << (first ? "=" : ":") << "overlap=" << overlap_;
        first = false;
    }
    if (gain_ != 1.00) {
        desc << (first ? "=" : ":") << "gain=" << gain_;
        first = false;
    }
    if (data_ != 0) {
        desc << (first ? "=" : ":") << "data=" << data_;
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
    if (fps_ != "auto") {
        desc << (first ? "=" : ":") << "fps=" << fps_;
        first = false;
    }
    if (legend_ != false) {
        desc << (first ? "=" : ":") << "legend=" << (legend_ ? "1" : "0");
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
