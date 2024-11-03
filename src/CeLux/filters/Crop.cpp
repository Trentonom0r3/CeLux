#include "Crop.hpp"
#include <sstream>

Crop::Crop(const std::string& out_w, const std::string& out_h, const std::string& xCropArea, const std::string& yCropArea, bool keep_aspect, bool exact) {
    // Initialize member variables from parameters
    this->out_w_ = out_w;
    this->out_h_ = out_h;
    this->xCropArea_ = xCropArea;
    this->yCropArea_ = yCropArea;
    this->keep_aspect_ = keep_aspect;
    this->exact_ = exact;
}

Crop::~Crop() {
    // Destructor implementation (if needed)
}

void Crop::setOut_w(const std::string& value) {
    out_w_ = value;
}

std::string Crop::getOut_w() const {
    return out_w_;
}

void Crop::setOut_h(const std::string& value) {
    out_h_ = value;
}

std::string Crop::getOut_h() const {
    return out_h_;
}

void Crop::setXCropArea(const std::string& value) {
    xCropArea_ = value;
}

std::string Crop::getXCropArea() const {
    return xCropArea_;
}

void Crop::setYCropArea(const std::string& value) {
    yCropArea_ = value;
}

std::string Crop::getYCropArea() const {
    return yCropArea_;
}

void Crop::setKeep_aspect(bool value) {
    keep_aspect_ = value;
}

bool Crop::getKeep_aspect() const {
    return keep_aspect_;
}

void Crop::setExact(bool value) {
    exact_ = value;
}

bool Crop::getExact() const {
    return exact_;
}

std::string Crop::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "crop";

    bool first = true;

    if (out_w_ != "iw") {
        desc << (first ? "=" : ":") << "out_w=" << out_w_;
        first = false;
    }
    if (out_h_ != "ih") {
        desc << (first ? "=" : ":") << "out_h=" << out_h_;
        first = false;
    }
    if (xCropArea_ != "(in_w-out_w)/2") {
        desc << (first ? "=" : ":") << "x=" << xCropArea_;
        first = false;
    }
    if (yCropArea_ != "(in_h-out_h)/2") {
        desc << (first ? "=" : ":") << "y=" << yCropArea_;
        first = false;
    }
    if (keep_aspect_ != false) {
        desc << (first ? "=" : ":") << "keep_aspect=" << (keep_aspect_ ? "1" : "0");
        first = false;
    }
    if (exact_ != false) {
        desc << (first ? "=" : ":") << "exact=" << (exact_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
