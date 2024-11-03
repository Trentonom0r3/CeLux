#include "Blackdetect.hpp"
#include <sstream>

Blackdetect::Blackdetect(double black_min_duration, double picture_black_ratio_th, double pixel_black_th) {
    // Initialize member variables from parameters
    this->black_min_duration_ = black_min_duration;
    this->picture_black_ratio_th_ = picture_black_ratio_th;
    this->pixel_black_th_ = pixel_black_th;
}

Blackdetect::~Blackdetect() {
    // Destructor implementation (if needed)
}

void Blackdetect::setBlack_min_duration(double value) {
    black_min_duration_ = value;
}

double Blackdetect::getBlack_min_duration() const {
    return black_min_duration_;
}

void Blackdetect::setPicture_black_ratio_th(double value) {
    picture_black_ratio_th_ = value;
}

double Blackdetect::getPicture_black_ratio_th() const {
    return picture_black_ratio_th_;
}

void Blackdetect::setPixel_black_th(double value) {
    pixel_black_th_ = value;
}

double Blackdetect::getPixel_black_th() const {
    return pixel_black_th_;
}

std::string Blackdetect::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "blackdetect";

    bool first = true;

    if (black_min_duration_ != 2.00) {
        desc << (first ? "=" : ":") << "black_min_duration=" << black_min_duration_;
        first = false;
    }
    if (picture_black_ratio_th_ != 0.98) {
        desc << (first ? "=" : ":") << "picture_black_ratio_th=" << picture_black_ratio_th_;
        first = false;
    }
    if (pixel_black_th_ != 0.10) {
        desc << (first ? "=" : ":") << "pixel_black_th=" << pixel_black_th_;
        first = false;
    }

    return desc.str();
}
