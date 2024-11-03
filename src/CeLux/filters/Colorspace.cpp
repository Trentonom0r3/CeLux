#include "Colorspace.hpp"
#include <sstream>

Colorspace::Colorspace(int all, int space, int range, int primaries, int trc, int format, bool fast, int dither, int wpadapt, int iall, int ispace, int irange, int iprimaries, int itrc) {
    // Initialize member variables from parameters
    this->all_ = all;
    this->space_ = space;
    this->range_ = range;
    this->primaries_ = primaries;
    this->trc_ = trc;
    this->format_ = format;
    this->fast_ = fast;
    this->dither_ = dither;
    this->wpadapt_ = wpadapt;
    this->iall_ = iall;
    this->ispace_ = ispace;
    this->irange_ = irange;
    this->iprimaries_ = iprimaries;
    this->itrc_ = itrc;
}

Colorspace::~Colorspace() {
    // Destructor implementation (if needed)
}

void Colorspace::setAll(int value) {
    all_ = value;
}

int Colorspace::getAll() const {
    return all_;
}

void Colorspace::setSpace(int value) {
    space_ = value;
}

int Colorspace::getSpace() const {
    return space_;
}

void Colorspace::setRange(int value) {
    range_ = value;
}

int Colorspace::getRange() const {
    return range_;
}

void Colorspace::setPrimaries(int value) {
    primaries_ = value;
}

int Colorspace::getPrimaries() const {
    return primaries_;
}

void Colorspace::setTrc(int value) {
    trc_ = value;
}

int Colorspace::getTrc() const {
    return trc_;
}

void Colorspace::setFormat(int value) {
    format_ = value;
}

int Colorspace::getFormat() const {
    return format_;
}

void Colorspace::setFast(bool value) {
    fast_ = value;
}

bool Colorspace::getFast() const {
    return fast_;
}

void Colorspace::setDither(int value) {
    dither_ = value;
}

int Colorspace::getDither() const {
    return dither_;
}

void Colorspace::setWpadapt(int value) {
    wpadapt_ = value;
}

int Colorspace::getWpadapt() const {
    return wpadapt_;
}

void Colorspace::setIall(int value) {
    iall_ = value;
}

int Colorspace::getIall() const {
    return iall_;
}

void Colorspace::setIspace(int value) {
    ispace_ = value;
}

int Colorspace::getIspace() const {
    return ispace_;
}

void Colorspace::setIrange(int value) {
    irange_ = value;
}

int Colorspace::getIrange() const {
    return irange_;
}

void Colorspace::setIprimaries(int value) {
    iprimaries_ = value;
}

int Colorspace::getIprimaries() const {
    return iprimaries_;
}

void Colorspace::setItrc(int value) {
    itrc_ = value;
}

int Colorspace::getItrc() const {
    return itrc_;
}

std::string Colorspace::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorspace";

    bool first = true;

    if (all_ != 0) {
        desc << (first ? "=" : ":") << "all=" << all_;
        first = false;
    }
    if (space_ != 2) {
        desc << (first ? "=" : ":") << "space=" << space_;
        first = false;
    }
    if (range_ != 0) {
        desc << (first ? "=" : ":") << "range=" << range_;
        first = false;
    }
    if (primaries_ != 2) {
        desc << (first ? "=" : ":") << "primaries=" << primaries_;
        first = false;
    }
    if (trc_ != 2) {
        desc << (first ? "=" : ":") << "trc=" << trc_;
        first = false;
    }
    if (format_ != -1) {
        desc << (first ? "=" : ":") << "format=" << format_;
        first = false;
    }
    if (fast_ != false) {
        desc << (first ? "=" : ":") << "fast=" << (fast_ ? "1" : "0");
        first = false;
    }
    if (dither_ != 0) {
        desc << (first ? "=" : ":") << "dither=" << dither_;
        first = false;
    }
    if (wpadapt_ != 0) {
        desc << (first ? "=" : ":") << "wpadapt=" << wpadapt_;
        first = false;
    }
    if (iall_ != 0) {
        desc << (first ? "=" : ":") << "iall=" << iall_;
        first = false;
    }
    if (ispace_ != 2) {
        desc << (first ? "=" : ":") << "ispace=" << ispace_;
        first = false;
    }
    if (irange_ != 0) {
        desc << (first ? "=" : ":") << "irange=" << irange_;
        first = false;
    }
    if (iprimaries_ != 2) {
        desc << (first ? "=" : ":") << "iprimaries=" << iprimaries_;
        first = false;
    }
    if (itrc_ != 2) {
        desc << (first ? "=" : ":") << "itrc=" << itrc_;
        first = false;
    }

    return desc.str();
}
