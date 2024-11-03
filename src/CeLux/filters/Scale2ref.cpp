#include "Scale2ref.hpp"
#include <sstream>

Scale2ref::Scale2ref(const std::string& width, const std::string& height, const std::string& flags, bool interl, const std::string& size, int in_color_matrix, int out_color_matrix, int in_range, int out_range, int in_v_chr_pos, int in_h_chr_pos, int out_v_chr_pos, int out_h_chr_pos, int force_original_aspect_ratio, int force_divisible_by, double param0, double param1, int eval) {
    // Initialize member variables from parameters
    this->width_ = width;
    this->height_ = height;
    this->flags_ = flags;
    this->interl_ = interl;
    this->size_ = size;
    this->in_color_matrix_ = in_color_matrix;
    this->out_color_matrix_ = out_color_matrix;
    this->in_range_ = in_range;
    this->out_range_ = out_range;
    this->in_v_chr_pos_ = in_v_chr_pos;
    this->in_h_chr_pos_ = in_h_chr_pos;
    this->out_v_chr_pos_ = out_v_chr_pos;
    this->out_h_chr_pos_ = out_h_chr_pos;
    this->force_original_aspect_ratio_ = force_original_aspect_ratio;
    this->force_divisible_by_ = force_divisible_by;
    this->param0_ = param0;
    this->param1_ = param1;
    this->eval_ = eval;
}

Scale2ref::~Scale2ref() {
    // Destructor implementation (if needed)
}

void Scale2ref::setWidth(const std::string& value) {
    width_ = value;
}

std::string Scale2ref::getWidth() const {
    return width_;
}

void Scale2ref::setHeight(const std::string& value) {
    height_ = value;
}

std::string Scale2ref::getHeight() const {
    return height_;
}

void Scale2ref::setFlags(const std::string& value) {
    flags_ = value;
}

std::string Scale2ref::getFlags() const {
    return flags_;
}

void Scale2ref::setInterl(bool value) {
    interl_ = value;
}

bool Scale2ref::getInterl() const {
    return interl_;
}

void Scale2ref::setSize(const std::string& value) {
    size_ = value;
}

std::string Scale2ref::getSize() const {
    return size_;
}

void Scale2ref::setIn_color_matrix(int value) {
    in_color_matrix_ = value;
}

int Scale2ref::getIn_color_matrix() const {
    return in_color_matrix_;
}

void Scale2ref::setOut_color_matrix(int value) {
    out_color_matrix_ = value;
}

int Scale2ref::getOut_color_matrix() const {
    return out_color_matrix_;
}

void Scale2ref::setIn_range(int value) {
    in_range_ = value;
}

int Scale2ref::getIn_range() const {
    return in_range_;
}

void Scale2ref::setOut_range(int value) {
    out_range_ = value;
}

int Scale2ref::getOut_range() const {
    return out_range_;
}

void Scale2ref::setIn_v_chr_pos(int value) {
    in_v_chr_pos_ = value;
}

int Scale2ref::getIn_v_chr_pos() const {
    return in_v_chr_pos_;
}

void Scale2ref::setIn_h_chr_pos(int value) {
    in_h_chr_pos_ = value;
}

int Scale2ref::getIn_h_chr_pos() const {
    return in_h_chr_pos_;
}

void Scale2ref::setOut_v_chr_pos(int value) {
    out_v_chr_pos_ = value;
}

int Scale2ref::getOut_v_chr_pos() const {
    return out_v_chr_pos_;
}

void Scale2ref::setOut_h_chr_pos(int value) {
    out_h_chr_pos_ = value;
}

int Scale2ref::getOut_h_chr_pos() const {
    return out_h_chr_pos_;
}

void Scale2ref::setForce_original_aspect_ratio(int value) {
    force_original_aspect_ratio_ = value;
}

int Scale2ref::getForce_original_aspect_ratio() const {
    return force_original_aspect_ratio_;
}

void Scale2ref::setForce_divisible_by(int value) {
    force_divisible_by_ = value;
}

int Scale2ref::getForce_divisible_by() const {
    return force_divisible_by_;
}

void Scale2ref::setParam0(double value) {
    param0_ = value;
}

double Scale2ref::getParam0() const {
    return param0_;
}

void Scale2ref::setParam1(double value) {
    param1_ = value;
}

double Scale2ref::getParam1() const {
    return param1_;
}

void Scale2ref::setEval(int value) {
    eval_ = value;
}

int Scale2ref::getEval() const {
    return eval_;
}

std::string Scale2ref::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "scale2ref";

    bool first = true;

    if (!width_.empty()) {
        desc << (first ? "=" : ":") << "width=" << width_;
        first = false;
    }
    if (!height_.empty()) {
        desc << (first ? "=" : ":") << "height=" << height_;
        first = false;
    }
    if (!flags_.empty()) {
        desc << (first ? "=" : ":") << "flags=" << flags_;
        first = false;
    }
    if (interl_ != false) {
        desc << (first ? "=" : ":") << "interl=" << (interl_ ? "1" : "0");
        first = false;
    }
    if (!size_.empty()) {
        desc << (first ? "=" : ":") << "size=" << size_;
        first = false;
    }
    if (in_color_matrix_ != -1) {
        desc << (first ? "=" : ":") << "in_color_matrix=" << in_color_matrix_;
        first = false;
    }
    if (out_color_matrix_ != 2) {
        desc << (first ? "=" : ":") << "out_color_matrix=" << out_color_matrix_;
        first = false;
    }
    if (in_range_ != 0) {
        desc << (first ? "=" : ":") << "in_range=" << in_range_;
        first = false;
    }
    if (out_range_ != 0) {
        desc << (first ? "=" : ":") << "out_range=" << out_range_;
        first = false;
    }
    if (in_v_chr_pos_ != -513) {
        desc << (first ? "=" : ":") << "in_v_chr_pos=" << in_v_chr_pos_;
        first = false;
    }
    if (in_h_chr_pos_ != -513) {
        desc << (first ? "=" : ":") << "in_h_chr_pos=" << in_h_chr_pos_;
        first = false;
    }
    if (out_v_chr_pos_ != -513) {
        desc << (first ? "=" : ":") << "out_v_chr_pos=" << out_v_chr_pos_;
        first = false;
    }
    if (out_h_chr_pos_ != -513) {
        desc << (first ? "=" : ":") << "out_h_chr_pos=" << out_h_chr_pos_;
        first = false;
    }
    if (force_original_aspect_ratio_ != 0) {
        desc << (first ? "=" : ":") << "force_original_aspect_ratio=" << force_original_aspect_ratio_;
        first = false;
    }
    if (force_divisible_by_ != 1) {
        desc << (first ? "=" : ":") << "force_divisible_by=" << force_divisible_by_;
        first = false;
    }
    if (param0_ != 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00) {
        desc << (first ? "=" : ":") << "param0=" << param0_;
        first = false;
    }
    if (param1_ != 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00) {
        desc << (first ? "=" : ":") << "param1=" << param1_;
        first = false;
    }
    if (eval_ != 0) {
        desc << (first ? "=" : ":") << "eval=" << eval_;
        first = false;
    }

    return desc.str();
}
