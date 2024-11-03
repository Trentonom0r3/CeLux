#include "Unsharp.hpp"
#include <sstream>

Unsharp::Unsharp(int luma_msize_x, int luma_msize_y, float luma_amount, int chroma_msize_x, int chroma_msize_y, float chroma_amount, int alpha_msize_x, int alpha_msize_y, float alpha_amount) {
    // Initialize member variables from parameters
    this->luma_msize_x_ = luma_msize_x;
    this->luma_msize_y_ = luma_msize_y;
    this->luma_amount_ = luma_amount;
    this->chroma_msize_x_ = chroma_msize_x;
    this->chroma_msize_y_ = chroma_msize_y;
    this->chroma_amount_ = chroma_amount;
    this->alpha_msize_x_ = alpha_msize_x;
    this->alpha_msize_y_ = alpha_msize_y;
    this->alpha_amount_ = alpha_amount;
}

Unsharp::~Unsharp() {
    // Destructor implementation (if needed)
}

void Unsharp::setLuma_msize_x(int value) {
    luma_msize_x_ = value;
}

int Unsharp::getLuma_msize_x() const {
    return luma_msize_x_;
}

void Unsharp::setLuma_msize_y(int value) {
    luma_msize_y_ = value;
}

int Unsharp::getLuma_msize_y() const {
    return luma_msize_y_;
}

void Unsharp::setLuma_amount(float value) {
    luma_amount_ = value;
}

float Unsharp::getLuma_amount() const {
    return luma_amount_;
}

void Unsharp::setChroma_msize_x(int value) {
    chroma_msize_x_ = value;
}

int Unsharp::getChroma_msize_x() const {
    return chroma_msize_x_;
}

void Unsharp::setChroma_msize_y(int value) {
    chroma_msize_y_ = value;
}

int Unsharp::getChroma_msize_y() const {
    return chroma_msize_y_;
}

void Unsharp::setChroma_amount(float value) {
    chroma_amount_ = value;
}

float Unsharp::getChroma_amount() const {
    return chroma_amount_;
}

void Unsharp::setAlpha_msize_x(int value) {
    alpha_msize_x_ = value;
}

int Unsharp::getAlpha_msize_x() const {
    return alpha_msize_x_;
}

void Unsharp::setAlpha_msize_y(int value) {
    alpha_msize_y_ = value;
}

int Unsharp::getAlpha_msize_y() const {
    return alpha_msize_y_;
}

void Unsharp::setAlpha_amount(float value) {
    alpha_amount_ = value;
}

float Unsharp::getAlpha_amount() const {
    return alpha_amount_;
}

std::string Unsharp::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "unsharp";

    bool first = true;

    if (luma_msize_x_ != 5) {
        desc << (first ? "=" : ":") << "luma_msize_x=" << luma_msize_x_;
        first = false;
    }
    if (luma_msize_y_ != 5) {
        desc << (first ? "=" : ":") << "luma_msize_y=" << luma_msize_y_;
        first = false;
    }
    if (luma_amount_ != 1.00) {
        desc << (first ? "=" : ":") << "luma_amount=" << luma_amount_;
        first = false;
    }
    if (chroma_msize_x_ != 5) {
        desc << (first ? "=" : ":") << "chroma_msize_x=" << chroma_msize_x_;
        first = false;
    }
    if (chroma_msize_y_ != 5) {
        desc << (first ? "=" : ":") << "chroma_msize_y=" << chroma_msize_y_;
        first = false;
    }
    if (chroma_amount_ != 0.00) {
        desc << (first ? "=" : ":") << "chroma_amount=" << chroma_amount_;
        first = false;
    }
    if (alpha_msize_x_ != 5) {
        desc << (first ? "=" : ":") << "alpha_msize_x=" << alpha_msize_x_;
        first = false;
    }
    if (alpha_msize_y_ != 5) {
        desc << (first ? "=" : ":") << "alpha_msize_y=" << alpha_msize_y_;
        first = false;
    }
    if (alpha_amount_ != 0.00) {
        desc << (first ? "=" : ":") << "alpha_amount=" << alpha_amount_;
        first = false;
    }

    return desc.str();
}
