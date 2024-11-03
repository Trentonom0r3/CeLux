#include "Showwavespic.hpp"
#include <sstream>

Showwavespic::Showwavespic(std::pair<int, int> size, bool split_channels, const std::string& colors, int scale, int draw, int filter) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->split_channels_ = split_channels;
    this->colors_ = colors;
    this->scale_ = scale;
    this->draw_ = draw;
    this->filter_ = filter;
}

Showwavespic::~Showwavespic() {
    // Destructor implementation (if needed)
}

void Showwavespic::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Showwavespic::getSize() const {
    return size_;
}

void Showwavespic::setSplit_channels(bool value) {
    split_channels_ = value;
}

bool Showwavespic::getSplit_channels() const {
    return split_channels_;
}

void Showwavespic::setColors(const std::string& value) {
    colors_ = value;
}

std::string Showwavespic::getColors() const {
    return colors_;
}

void Showwavespic::setScale(int value) {
    scale_ = value;
}

int Showwavespic::getScale() const {
    return scale_;
}

void Showwavespic::setDraw(int value) {
    draw_ = value;
}

int Showwavespic::getDraw() const {
    return draw_;
}

void Showwavespic::setFilter(int value) {
    filter_ = value;
}

int Showwavespic::getFilter() const {
    return filter_;
}

std::string Showwavespic::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showwavespic";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (split_channels_ != false) {
        desc << (first ? "=" : ":") << "split_channels=" << (split_channels_ ? "1" : "0");
        first = false;
    }
    if (colors_ != "red|green|blue|yellow|orange|lime|pink|magenta|brown") {
        desc << (first ? "=" : ":") << "colors=" << colors_;
        first = false;
    }
    if (scale_ != 0) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (draw_ != 0) {
        desc << (first ? "=" : ":") << "draw=" << draw_;
        first = false;
    }
    if (filter_ != 0) {
        desc << (first ? "=" : ":") << "filter=" << filter_;
        first = false;
    }

    return desc.str();
}
