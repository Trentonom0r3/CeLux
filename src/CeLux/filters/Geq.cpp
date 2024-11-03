#include "Geq.hpp"
#include <sstream>

Geq::Geq(const std::string& lum_expr, const std::string& cb_expr, const std::string& cr_expr, const std::string& alpha_expr, const std::string& red_expr, const std::string& green_expr, const std::string& blue_expr, int interpolation) {
    // Initialize member variables from parameters
    this->lum_expr_ = lum_expr;
    this->cb_expr_ = cb_expr;
    this->cr_expr_ = cr_expr;
    this->alpha_expr_ = alpha_expr;
    this->red_expr_ = red_expr;
    this->green_expr_ = green_expr;
    this->blue_expr_ = blue_expr;
    this->interpolation_ = interpolation;
}

Geq::~Geq() {
    // Destructor implementation (if needed)
}

void Geq::setLum_expr(const std::string& value) {
    lum_expr_ = value;
}

std::string Geq::getLum_expr() const {
    return lum_expr_;
}

void Geq::setCb_expr(const std::string& value) {
    cb_expr_ = value;
}

std::string Geq::getCb_expr() const {
    return cb_expr_;
}

void Geq::setCr_expr(const std::string& value) {
    cr_expr_ = value;
}

std::string Geq::getCr_expr() const {
    return cr_expr_;
}

void Geq::setAlpha_expr(const std::string& value) {
    alpha_expr_ = value;
}

std::string Geq::getAlpha_expr() const {
    return alpha_expr_;
}

void Geq::setRed_expr(const std::string& value) {
    red_expr_ = value;
}

std::string Geq::getRed_expr() const {
    return red_expr_;
}

void Geq::setGreen_expr(const std::string& value) {
    green_expr_ = value;
}

std::string Geq::getGreen_expr() const {
    return green_expr_;
}

void Geq::setBlue_expr(const std::string& value) {
    blue_expr_ = value;
}

std::string Geq::getBlue_expr() const {
    return blue_expr_;
}

void Geq::setInterpolation(int value) {
    interpolation_ = value;
}

int Geq::getInterpolation() const {
    return interpolation_;
}

std::string Geq::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "geq";

    bool first = true;

    if (!lum_expr_.empty()) {
        desc << (first ? "=" : ":") << "lum_expr=" << lum_expr_;
        first = false;
    }
    if (!cb_expr_.empty()) {
        desc << (first ? "=" : ":") << "cb_expr=" << cb_expr_;
        first = false;
    }
    if (!cr_expr_.empty()) {
        desc << (first ? "=" : ":") << "cr_expr=" << cr_expr_;
        first = false;
    }
    if (!alpha_expr_.empty()) {
        desc << (first ? "=" : ":") << "alpha_expr=" << alpha_expr_;
        first = false;
    }
    if (!red_expr_.empty()) {
        desc << (first ? "=" : ":") << "red_expr=" << red_expr_;
        first = false;
    }
    if (!green_expr_.empty()) {
        desc << (first ? "=" : ":") << "green_expr=" << green_expr_;
        first = false;
    }
    if (!blue_expr_.empty()) {
        desc << (first ? "=" : ":") << "blue_expr=" << blue_expr_;
        first = false;
    }
    if (interpolation_ != 1) {
        desc << (first ? "=" : ":") << "interpolation=" << interpolation_;
        first = false;
    }

    return desc.str();
}
