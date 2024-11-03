#include "Il.hpp"
#include <sstream>

Il::Il(int luma_mode, int chroma_mode, int alpha_mode, bool luma_swap, bool chroma_swap, bool alpha_swap) {
    // Initialize member variables from parameters
    this->luma_mode_ = luma_mode;
    this->chroma_mode_ = chroma_mode;
    this->alpha_mode_ = alpha_mode;
    this->luma_swap_ = luma_swap;
    this->chroma_swap_ = chroma_swap;
    this->alpha_swap_ = alpha_swap;
}

Il::~Il() {
    // Destructor implementation (if needed)
}

void Il::setLuma_mode(int value) {
    luma_mode_ = value;
}

int Il::getLuma_mode() const {
    return luma_mode_;
}

void Il::setChroma_mode(int value) {
    chroma_mode_ = value;
}

int Il::getChroma_mode() const {
    return chroma_mode_;
}

void Il::setAlpha_mode(int value) {
    alpha_mode_ = value;
}

int Il::getAlpha_mode() const {
    return alpha_mode_;
}

void Il::setLuma_swap(bool value) {
    luma_swap_ = value;
}

bool Il::getLuma_swap() const {
    return luma_swap_;
}

void Il::setChroma_swap(bool value) {
    chroma_swap_ = value;
}

bool Il::getChroma_swap() const {
    return chroma_swap_;
}

void Il::setAlpha_swap(bool value) {
    alpha_swap_ = value;
}

bool Il::getAlpha_swap() const {
    return alpha_swap_;
}

std::string Il::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "il";

    bool first = true;

    if (luma_mode_ != 0) {
        desc << (first ? "=" : ":") << "luma_mode=" << luma_mode_;
        first = false;
    }
    if (chroma_mode_ != 0) {
        desc << (first ? "=" : ":") << "chroma_mode=" << chroma_mode_;
        first = false;
    }
    if (alpha_mode_ != 0) {
        desc << (first ? "=" : ":") << "alpha_mode=" << alpha_mode_;
        first = false;
    }
    if (luma_swap_ != false) {
        desc << (first ? "=" : ":") << "luma_swap=" << (luma_swap_ ? "1" : "0");
        first = false;
    }
    if (chroma_swap_ != false) {
        desc << (first ? "=" : ":") << "chroma_swap=" << (chroma_swap_ ? "1" : "0");
        first = false;
    }
    if (alpha_swap_ != false) {
        desc << (first ? "=" : ":") << "alpha_swap=" << (alpha_swap_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
