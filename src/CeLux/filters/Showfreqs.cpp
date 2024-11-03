#include "Showfreqs.hpp"
#include <sstream>

Showfreqs::Showfreqs(std::pair<int, int> size, std::pair<int, int> rate, int mode, int ascale, int fscale, int win_size, int win_func, float overlap, int averaging, const std::string& colors, int cmode, float minamp, int data, const std::string& channels) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->mode_ = mode;
    this->ascale_ = ascale;
    this->fscale_ = fscale;
    this->win_size_ = win_size;
    this->win_func_ = win_func;
    this->overlap_ = overlap;
    this->averaging_ = averaging;
    this->colors_ = colors;
    this->cmode_ = cmode;
    this->minamp_ = minamp;
    this->data_ = data;
    this->channels_ = channels;
}

Showfreqs::~Showfreqs() {
    // Destructor implementation (if needed)
}

void Showfreqs::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Showfreqs::getSize() const {
    return size_;
}

void Showfreqs::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Showfreqs::getRate() const {
    return rate_;
}

void Showfreqs::setMode(int value) {
    mode_ = value;
}

int Showfreqs::getMode() const {
    return mode_;
}

void Showfreqs::setAscale(int value) {
    ascale_ = value;
}

int Showfreqs::getAscale() const {
    return ascale_;
}

void Showfreqs::setFscale(int value) {
    fscale_ = value;
}

int Showfreqs::getFscale() const {
    return fscale_;
}

void Showfreqs::setWin_size(int value) {
    win_size_ = value;
}

int Showfreqs::getWin_size() const {
    return win_size_;
}

void Showfreqs::setWin_func(int value) {
    win_func_ = value;
}

int Showfreqs::getWin_func() const {
    return win_func_;
}

void Showfreqs::setOverlap(float value) {
    overlap_ = value;
}

float Showfreqs::getOverlap() const {
    return overlap_;
}

void Showfreqs::setAveraging(int value) {
    averaging_ = value;
}

int Showfreqs::getAveraging() const {
    return averaging_;
}

void Showfreqs::setColors(const std::string& value) {
    colors_ = value;
}

std::string Showfreqs::getColors() const {
    return colors_;
}

void Showfreqs::setCmode(int value) {
    cmode_ = value;
}

int Showfreqs::getCmode() const {
    return cmode_;
}

void Showfreqs::setMinamp(float value) {
    minamp_ = value;
}

float Showfreqs::getMinamp() const {
    return minamp_;
}

void Showfreqs::setData(int value) {
    data_ = value;
}

int Showfreqs::getData() const {
    return data_;
}

void Showfreqs::setChannels(const std::string& value) {
    channels_ = value;
}

std::string Showfreqs::getChannels() const {
    return channels_;
}

std::string Showfreqs::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showfreqs";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (mode_ != 1) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (ascale_ != 3) {
        desc << (first ? "=" : ":") << "ascale=" << ascale_;
        first = false;
    }
    if (fscale_ != 0) {
        desc << (first ? "=" : ":") << "fscale=" << fscale_;
        first = false;
    }
    if (win_size_ != 2048) {
        desc << (first ? "=" : ":") << "win_size=" << win_size_;
        first = false;
    }
    if (win_func_ != 1) {
        desc << (first ? "=" : ":") << "win_func=" << win_func_;
        first = false;
    }
    if (overlap_ != 1.00) {
        desc << (first ? "=" : ":") << "overlap=" << overlap_;
        first = false;
    }
    if (averaging_ != 1) {
        desc << (first ? "=" : ":") << "averaging=" << averaging_;
        first = false;
    }
    if (colors_ != "red|green|blue|yellow|orange|lime|pink|magenta|brown") {
        desc << (first ? "=" : ":") << "colors=" << colors_;
        first = false;
    }
    if (cmode_ != 0) {
        desc << (first ? "=" : ":") << "cmode=" << cmode_;
        first = false;
    }
    if (minamp_ != 0.00) {
        desc << (first ? "=" : ":") << "minamp=" << minamp_;
        first = false;
    }
    if (data_ != 0) {
        desc << (first ? "=" : ":") << "data=" << data_;
        first = false;
    }
    if (channels_ != "all") {
        desc << (first ? "=" : ":") << "channels=" << channels_;
        first = false;
    }

    return desc.str();
}
