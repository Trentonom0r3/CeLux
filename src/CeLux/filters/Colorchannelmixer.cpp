#include "Colorchannelmixer.hpp"
#include <sstream>

Colorchannelmixer::Colorchannelmixer(double rr, double rg, double rb, double ra, double gr, double gg, double gb, double ga, double br, double bg, double bb, double ba, double ar, double ag, double ab, double aa, int pc, double pa) {
    // Initialize member variables from parameters
    this->rr_ = rr;
    this->rg_ = rg;
    this->rb_ = rb;
    this->ra_ = ra;
    this->gr_ = gr;
    this->gg_ = gg;
    this->gb_ = gb;
    this->ga_ = ga;
    this->br_ = br;
    this->bg_ = bg;
    this->bb_ = bb;
    this->ba_ = ba;
    this->ar_ = ar;
    this->ag_ = ag;
    this->ab_ = ab;
    this->aa_ = aa;
    this->pc_ = pc;
    this->pa_ = pa;
}

Colorchannelmixer::~Colorchannelmixer() {
    // Destructor implementation (if needed)
}

void Colorchannelmixer::setRr(double value) {
    rr_ = value;
}

double Colorchannelmixer::getRr() const {
    return rr_;
}

void Colorchannelmixer::setRg(double value) {
    rg_ = value;
}

double Colorchannelmixer::getRg() const {
    return rg_;
}

void Colorchannelmixer::setRb(double value) {
    rb_ = value;
}

double Colorchannelmixer::getRb() const {
    return rb_;
}

void Colorchannelmixer::setRa(double value) {
    ra_ = value;
}

double Colorchannelmixer::getRa() const {
    return ra_;
}

void Colorchannelmixer::setGr(double value) {
    gr_ = value;
}

double Colorchannelmixer::getGr() const {
    return gr_;
}

void Colorchannelmixer::setGg(double value) {
    gg_ = value;
}

double Colorchannelmixer::getGg() const {
    return gg_;
}

void Colorchannelmixer::setGb(double value) {
    gb_ = value;
}

double Colorchannelmixer::getGb() const {
    return gb_;
}

void Colorchannelmixer::setGa(double value) {
    ga_ = value;
}

double Colorchannelmixer::getGa() const {
    return ga_;
}

void Colorchannelmixer::setBr(double value) {
    br_ = value;
}

double Colorchannelmixer::getBr() const {
    return br_;
}

void Colorchannelmixer::setBg(double value) {
    bg_ = value;
}

double Colorchannelmixer::getBg() const {
    return bg_;
}

void Colorchannelmixer::setBb(double value) {
    bb_ = value;
}

double Colorchannelmixer::getBb() const {
    return bb_;
}

void Colorchannelmixer::setBa(double value) {
    ba_ = value;
}

double Colorchannelmixer::getBa() const {
    return ba_;
}

void Colorchannelmixer::setAr(double value) {
    ar_ = value;
}

double Colorchannelmixer::getAr() const {
    return ar_;
}

void Colorchannelmixer::setAg(double value) {
    ag_ = value;
}

double Colorchannelmixer::getAg() const {
    return ag_;
}

void Colorchannelmixer::setAb(double value) {
    ab_ = value;
}

double Colorchannelmixer::getAb() const {
    return ab_;
}

void Colorchannelmixer::setAa(double value) {
    aa_ = value;
}

double Colorchannelmixer::getAa() const {
    return aa_;
}

void Colorchannelmixer::setPc(int value) {
    pc_ = value;
}

int Colorchannelmixer::getPc() const {
    return pc_;
}

void Colorchannelmixer::setPa(double value) {
    pa_ = value;
}

double Colorchannelmixer::getPa() const {
    return pa_;
}

std::string Colorchannelmixer::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorchannelmixer";

    bool first = true;

    if (rr_ != 1.00) {
        desc << (first ? "=" : ":") << "rr=" << rr_;
        first = false;
    }
    if (rg_ != 0.00) {
        desc << (first ? "=" : ":") << "rg=" << rg_;
        first = false;
    }
    if (rb_ != 0.00) {
        desc << (first ? "=" : ":") << "rb=" << rb_;
        first = false;
    }
    if (ra_ != 0.00) {
        desc << (first ? "=" : ":") << "ra=" << ra_;
        first = false;
    }
    if (gr_ != 0.00) {
        desc << (first ? "=" : ":") << "gr=" << gr_;
        first = false;
    }
    if (gg_ != 1.00) {
        desc << (first ? "=" : ":") << "gg=" << gg_;
        first = false;
    }
    if (gb_ != 0.00) {
        desc << (first ? "=" : ":") << "gb=" << gb_;
        first = false;
    }
    if (ga_ != 0.00) {
        desc << (first ? "=" : ":") << "ga=" << ga_;
        first = false;
    }
    if (br_ != 0.00) {
        desc << (first ? "=" : ":") << "br=" << br_;
        first = false;
    }
    if (bg_ != 0.00) {
        desc << (first ? "=" : ":") << "bg=" << bg_;
        first = false;
    }
    if (bb_ != 1.00) {
        desc << (first ? "=" : ":") << "bb=" << bb_;
        first = false;
    }
    if (ba_ != 0.00) {
        desc << (first ? "=" : ":") << "ba=" << ba_;
        first = false;
    }
    if (ar_ != 0.00) {
        desc << (first ? "=" : ":") << "ar=" << ar_;
        first = false;
    }
    if (ag_ != 0.00) {
        desc << (first ? "=" : ":") << "ag=" << ag_;
        first = false;
    }
    if (ab_ != 0.00) {
        desc << (first ? "=" : ":") << "ab=" << ab_;
        first = false;
    }
    if (aa_ != 1.00) {
        desc << (first ? "=" : ":") << "aa=" << aa_;
        first = false;
    }
    if (pc_ != 0) {
        desc << (first ? "=" : ":") << "pc=" << pc_;
        first = false;
    }
    if (pa_ != 0.00) {
        desc << (first ? "=" : ":") << "pa=" << pa_;
        first = false;
    }

    return desc.str();
}
