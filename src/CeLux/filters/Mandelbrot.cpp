#include "Mandelbrot.hpp"
#include <sstream>

Mandelbrot::Mandelbrot(std::pair<int, int> size, std::pair<int, int> rate, int maxiter, double start_x, double start_y, double start_scale, double end_scale, double end_pts, double bailout, double morphxf, double morphyf, double morphamp, int outer, int inner) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->maxiter_ = maxiter;
    this->start_x_ = start_x;
    this->start_y_ = start_y;
    this->start_scale_ = start_scale;
    this->end_scale_ = end_scale;
    this->end_pts_ = end_pts;
    this->bailout_ = bailout;
    this->morphxf_ = morphxf;
    this->morphyf_ = morphyf;
    this->morphamp_ = morphamp;
    this->outer_ = outer;
    this->inner_ = inner;
}

Mandelbrot::~Mandelbrot() {
    // Destructor implementation (if needed)
}

void Mandelbrot::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Mandelbrot::getSize() const {
    return size_;
}

void Mandelbrot::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Mandelbrot::getRate() const {
    return rate_;
}

void Mandelbrot::setMaxiter(int value) {
    maxiter_ = value;
}

int Mandelbrot::getMaxiter() const {
    return maxiter_;
}

void Mandelbrot::setStart_x(double value) {
    start_x_ = value;
}

double Mandelbrot::getStart_x() const {
    return start_x_;
}

void Mandelbrot::setStart_y(double value) {
    start_y_ = value;
}

double Mandelbrot::getStart_y() const {
    return start_y_;
}

void Mandelbrot::setStart_scale(double value) {
    start_scale_ = value;
}

double Mandelbrot::getStart_scale() const {
    return start_scale_;
}

void Mandelbrot::setEnd_scale(double value) {
    end_scale_ = value;
}

double Mandelbrot::getEnd_scale() const {
    return end_scale_;
}

void Mandelbrot::setEnd_pts(double value) {
    end_pts_ = value;
}

double Mandelbrot::getEnd_pts() const {
    return end_pts_;
}

void Mandelbrot::setBailout(double value) {
    bailout_ = value;
}

double Mandelbrot::getBailout() const {
    return bailout_;
}

void Mandelbrot::setMorphxf(double value) {
    morphxf_ = value;
}

double Mandelbrot::getMorphxf() const {
    return morphxf_;
}

void Mandelbrot::setMorphyf(double value) {
    morphyf_ = value;
}

double Mandelbrot::getMorphyf() const {
    return morphyf_;
}

void Mandelbrot::setMorphamp(double value) {
    morphamp_ = value;
}

double Mandelbrot::getMorphamp() const {
    return morphamp_;
}

void Mandelbrot::setOuter(int value) {
    outer_ = value;
}

int Mandelbrot::getOuter() const {
    return outer_;
}

void Mandelbrot::setInner(int value) {
    inner_ = value;
}

int Mandelbrot::getInner() const {
    return inner_;
}

std::string Mandelbrot::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "mandelbrot";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (maxiter_ != 7189) {
        desc << (first ? "=" : ":") << "maxiter=" << maxiter_;
        first = false;
    }
    if (start_x_ != -0.74) {
        desc << (first ? "=" : ":") << "start_x=" << start_x_;
        first = false;
    }
    if (start_y_ != -0.13) {
        desc << (first ? "=" : ":") << "start_y=" << start_y_;
        first = false;
    }
    if (start_scale_ != 3.00) {
        desc << (first ? "=" : ":") << "start_scale=" << start_scale_;
        first = false;
    }
    if (end_scale_ != 0.30) {
        desc << (first ? "=" : ":") << "end_scale=" << end_scale_;
        first = false;
    }
    if (end_pts_ != 400.00) {
        desc << (first ? "=" : ":") << "end_pts=" << end_pts_;
        first = false;
    }
    if (bailout_ != 10.00) {
        desc << (first ? "=" : ":") << "bailout=" << bailout_;
        first = false;
    }
    if (morphxf_ != 0.01) {
        desc << (first ? "=" : ":") << "morphxf=" << morphxf_;
        first = false;
    }
    if (morphyf_ != 0.01) {
        desc << (first ? "=" : ":") << "morphyf=" << morphyf_;
        first = false;
    }
    if (morphamp_ != 0.00) {
        desc << (first ? "=" : ":") << "morphamp=" << morphamp_;
        first = false;
    }
    if (outer_ != 1) {
        desc << (first ? "=" : ":") << "outer=" << outer_;
        first = false;
    }
    if (inner_ != 3) {
        desc << (first ? "=" : ":") << "inner=" << inner_;
        first = false;
    }

    return desc.str();
}
