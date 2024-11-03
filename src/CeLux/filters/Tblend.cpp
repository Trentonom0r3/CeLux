#include "Tblend.hpp"
#include <sstream>

Tblend::Tblend(int c0_mode, int c1_mode, int c2_mode, int c3_mode, int all_mode, const std::string& c0_expr, const std::string& c1_expr, const std::string& c2_expr, const std::string& c3_expr, const std::string& all_expr, double c0_opacity, double c1_opacity, double c2_opacity, double c3_opacity, double all_opacity) {
    // Initialize member variables from parameters
    this->c0_mode_ = c0_mode;
    this->c1_mode_ = c1_mode;
    this->c2_mode_ = c2_mode;
    this->c3_mode_ = c3_mode;
    this->all_mode_ = all_mode;
    this->c0_expr_ = c0_expr;
    this->c1_expr_ = c1_expr;
    this->c2_expr_ = c2_expr;
    this->c3_expr_ = c3_expr;
    this->all_expr_ = all_expr;
    this->c0_opacity_ = c0_opacity;
    this->c1_opacity_ = c1_opacity;
    this->c2_opacity_ = c2_opacity;
    this->c3_opacity_ = c3_opacity;
    this->all_opacity_ = all_opacity;
}

Tblend::~Tblend() {
    // Destructor implementation (if needed)
}

void Tblend::setC0_mode(int value) {
    c0_mode_ = value;
}

int Tblend::getC0_mode() const {
    return c0_mode_;
}

void Tblend::setC1_mode(int value) {
    c1_mode_ = value;
}

int Tblend::getC1_mode() const {
    return c1_mode_;
}

void Tblend::setC2_mode(int value) {
    c2_mode_ = value;
}

int Tblend::getC2_mode() const {
    return c2_mode_;
}

void Tblend::setC3_mode(int value) {
    c3_mode_ = value;
}

int Tblend::getC3_mode() const {
    return c3_mode_;
}

void Tblend::setAll_mode(int value) {
    all_mode_ = value;
}

int Tblend::getAll_mode() const {
    return all_mode_;
}

void Tblend::setC0_expr(const std::string& value) {
    c0_expr_ = value;
}

std::string Tblend::getC0_expr() const {
    return c0_expr_;
}

void Tblend::setC1_expr(const std::string& value) {
    c1_expr_ = value;
}

std::string Tblend::getC1_expr() const {
    return c1_expr_;
}

void Tblend::setC2_expr(const std::string& value) {
    c2_expr_ = value;
}

std::string Tblend::getC2_expr() const {
    return c2_expr_;
}

void Tblend::setC3_expr(const std::string& value) {
    c3_expr_ = value;
}

std::string Tblend::getC3_expr() const {
    return c3_expr_;
}

void Tblend::setAll_expr(const std::string& value) {
    all_expr_ = value;
}

std::string Tblend::getAll_expr() const {
    return all_expr_;
}

void Tblend::setC0_opacity(double value) {
    c0_opacity_ = value;
}

double Tblend::getC0_opacity() const {
    return c0_opacity_;
}

void Tblend::setC1_opacity(double value) {
    c1_opacity_ = value;
}

double Tblend::getC1_opacity() const {
    return c1_opacity_;
}

void Tblend::setC2_opacity(double value) {
    c2_opacity_ = value;
}

double Tblend::getC2_opacity() const {
    return c2_opacity_;
}

void Tblend::setC3_opacity(double value) {
    c3_opacity_ = value;
}

double Tblend::getC3_opacity() const {
    return c3_opacity_;
}

void Tblend::setAll_opacity(double value) {
    all_opacity_ = value;
}

double Tblend::getAll_opacity() const {
    return all_opacity_;
}

std::string Tblend::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "tblend";

    bool first = true;

    if (c0_mode_ != 0) {
        desc << (first ? "=" : ":") << "c0_mode=" << c0_mode_;
        first = false;
    }
    if (c1_mode_ != 0) {
        desc << (first ? "=" : ":") << "c1_mode=" << c1_mode_;
        first = false;
    }
    if (c2_mode_ != 0) {
        desc << (first ? "=" : ":") << "c2_mode=" << c2_mode_;
        first = false;
    }
    if (c3_mode_ != 0) {
        desc << (first ? "=" : ":") << "c3_mode=" << c3_mode_;
        first = false;
    }
    if (all_mode_ != -1) {
        desc << (first ? "=" : ":") << "all_mode=" << all_mode_;
        first = false;
    }
    if (!c0_expr_.empty()) {
        desc << (first ? "=" : ":") << "c0_expr=" << c0_expr_;
        first = false;
    }
    if (!c1_expr_.empty()) {
        desc << (first ? "=" : ":") << "c1_expr=" << c1_expr_;
        first = false;
    }
    if (!c2_expr_.empty()) {
        desc << (first ? "=" : ":") << "c2_expr=" << c2_expr_;
        first = false;
    }
    if (!c3_expr_.empty()) {
        desc << (first ? "=" : ":") << "c3_expr=" << c3_expr_;
        first = false;
    }
    if (!all_expr_.empty()) {
        desc << (first ? "=" : ":") << "all_expr=" << all_expr_;
        first = false;
    }
    if (c0_opacity_ != 1.00) {
        desc << (first ? "=" : ":") << "c0_opacity=" << c0_opacity_;
        first = false;
    }
    if (c1_opacity_ != 1.00) {
        desc << (first ? "=" : ":") << "c1_opacity=" << c1_opacity_;
        first = false;
    }
    if (c2_opacity_ != 1.00) {
        desc << (first ? "=" : ":") << "c2_opacity=" << c2_opacity_;
        first = false;
    }
    if (c3_opacity_ != 1.00) {
        desc << (first ? "=" : ":") << "c3_opacity=" << c3_opacity_;
        first = false;
    }
    if (all_opacity_ != 1.00) {
        desc << (first ? "=" : ":") << "all_opacity=" << all_opacity_;
        first = false;
    }

    return desc.str();
}
