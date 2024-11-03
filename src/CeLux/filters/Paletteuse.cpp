#include "Paletteuse.hpp"
#include <sstream>

Paletteuse::Paletteuse(int dither, int bayer_scale, int diff_mode, bool new_, int alpha_threshold, const std::string& debug_kdtree) {
    // Initialize member variables from parameters
    this->dither_ = dither;
    this->bayer_scale_ = bayer_scale;
    this->diff_mode_ = diff_mode;
    this->new__ = new_;
    this->alpha_threshold_ = alpha_threshold;
    this->debug_kdtree_ = debug_kdtree;
}

Paletteuse::~Paletteuse() {
    // Destructor implementation (if needed)
}

void Paletteuse::setDither(int value) {
    dither_ = value;
}

int Paletteuse::getDither() const {
    return dither_;
}

void Paletteuse::setBayer_scale(int value) {
    bayer_scale_ = value;
}

int Paletteuse::getBayer_scale() const {
    return bayer_scale_;
}

void Paletteuse::setDiff_mode(int value) {
    diff_mode_ = value;
}

int Paletteuse::getDiff_mode() const {
    return diff_mode_;
}

void Paletteuse::setNew_(bool value) {
    new__ = value;
}

bool Paletteuse::getNew_() const {
    return new__;
}

void Paletteuse::setAlpha_threshold(int value) {
    alpha_threshold_ = value;
}

int Paletteuse::getAlpha_threshold() const {
    return alpha_threshold_;
}

void Paletteuse::setDebug_kdtree(const std::string& value) {
    debug_kdtree_ = value;
}

std::string Paletteuse::getDebug_kdtree() const {
    return debug_kdtree_;
}

std::string Paletteuse::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "paletteuse";

    bool first = true;

    if (dither_ != 5) {
        desc << (first ? "=" : ":") << "dither=" << dither_;
        first = false;
    }
    if (bayer_scale_ != 2) {
        desc << (first ? "=" : ":") << "bayer_scale=" << bayer_scale_;
        first = false;
    }
    if (diff_mode_ != 0) {
        desc << (first ? "=" : ":") << "diff_mode=" << diff_mode_;
        first = false;
    }
    if (new__ != false) {
        desc << (first ? "=" : ":") << "new=" << (new__ ? "1" : "0");
        first = false;
    }
    if (alpha_threshold_ != 128) {
        desc << (first ? "=" : ":") << "alpha_threshold=" << alpha_threshold_;
        first = false;
    }
    if (!debug_kdtree_.empty()) {
        desc << (first ? "=" : ":") << "debug_kdtree=" << debug_kdtree_;
        first = false;
    }

    return desc.str();
}
