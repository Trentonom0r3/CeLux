#include "Avectorscope.hpp"
#include <sstream>

Avectorscope::Avectorscope(int mode, std::pair<int, int> rate, std::pair<int, int> size, int rc, int gc, int bc, int ac, int rf, int gf, int bf, int af, double zoom, int draw, int scale, bool swap, int mirror) {
    // Initialize member variables from parameters
    this->mode_ = mode;
    this->rate_ = rate;
    this->size_ = size;
    this->rc_ = rc;
    this->gc_ = gc;
    this->bc_ = bc;
    this->ac_ = ac;
    this->rf_ = rf;
    this->gf_ = gf;
    this->bf_ = bf;
    this->af_ = af;
    this->zoom_ = zoom;
    this->draw_ = draw;
    this->scale_ = scale;
    this->swap_ = swap;
    this->mirror_ = mirror;
}

Avectorscope::~Avectorscope() {
    // Destructor implementation (if needed)
}

void Avectorscope::setMode(int value) {
    mode_ = value;
}

int Avectorscope::getMode() const {
    return mode_;
}

void Avectorscope::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Avectorscope::getRate() const {
    return rate_;
}

void Avectorscope::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Avectorscope::getSize() const {
    return size_;
}

void Avectorscope::setRc(int value) {
    rc_ = value;
}

int Avectorscope::getRc() const {
    return rc_;
}

void Avectorscope::setGc(int value) {
    gc_ = value;
}

int Avectorscope::getGc() const {
    return gc_;
}

void Avectorscope::setBc(int value) {
    bc_ = value;
}

int Avectorscope::getBc() const {
    return bc_;
}

void Avectorscope::setAc(int value) {
    ac_ = value;
}

int Avectorscope::getAc() const {
    return ac_;
}

void Avectorscope::setRf(int value) {
    rf_ = value;
}

int Avectorscope::getRf() const {
    return rf_;
}

void Avectorscope::setGf(int value) {
    gf_ = value;
}

int Avectorscope::getGf() const {
    return gf_;
}

void Avectorscope::setBf(int value) {
    bf_ = value;
}

int Avectorscope::getBf() const {
    return bf_;
}

void Avectorscope::setAf(int value) {
    af_ = value;
}

int Avectorscope::getAf() const {
    return af_;
}

void Avectorscope::setZoom(double value) {
    zoom_ = value;
}

double Avectorscope::getZoom() const {
    return zoom_;
}

void Avectorscope::setDraw(int value) {
    draw_ = value;
}

int Avectorscope::getDraw() const {
    return draw_;
}

void Avectorscope::setScale(int value) {
    scale_ = value;
}

int Avectorscope::getScale() const {
    return scale_;
}

void Avectorscope::setSwap(bool value) {
    swap_ = value;
}

bool Avectorscope::getSwap() const {
    return swap_;
}

void Avectorscope::setMirror(int value) {
    mirror_ = value;
}

int Avectorscope::getMirror() const {
    return mirror_;
}

std::string Avectorscope::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "avectorscope";

    bool first = true;

    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rc_ != 40) {
        desc << (first ? "=" : ":") << "rc=" << rc_;
        first = false;
    }
    if (gc_ != 160) {
        desc << (first ? "=" : ":") << "gc=" << gc_;
        first = false;
    }
    if (bc_ != 80) {
        desc << (first ? "=" : ":") << "bc=" << bc_;
        first = false;
    }
    if (ac_ != 255) {
        desc << (first ? "=" : ":") << "ac=" << ac_;
        first = false;
    }
    if (rf_ != 15) {
        desc << (first ? "=" : ":") << "rf=" << rf_;
        first = false;
    }
    if (gf_ != 10) {
        desc << (first ? "=" : ":") << "gf=" << gf_;
        first = false;
    }
    if (bf_ != 5) {
        desc << (first ? "=" : ":") << "bf=" << bf_;
        first = false;
    }
    if (af_ != 5) {
        desc << (first ? "=" : ":") << "af=" << af_;
        first = false;
    }
    if (zoom_ != 1.00) {
        desc << (first ? "=" : ":") << "zoom=" << zoom_;
        first = false;
    }
    if (draw_ != 0) {
        desc << (first ? "=" : ":") << "draw=" << draw_;
        first = false;
    }
    if (scale_ != 0) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (swap_ != true) {
        desc << (first ? "=" : ":") << "swap=" << (swap_ ? "1" : "0");
        first = false;
    }
    if (mirror_ != 0) {
        desc << (first ? "=" : ":") << "mirror=" << mirror_;
        first = false;
    }

    return desc.str();
}
