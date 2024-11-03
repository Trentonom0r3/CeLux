#include "Colorlevels.hpp"
#include <sstream>

Colorlevels::Colorlevels(double rimin, double gimin, double bimin, double aimin, double rimax, double gimax, double bimax, double aimax, double romin, double gomin, double bomin, double aomin, double romax, double gomax, double bomax, double aomax, int preserve) {
    // Initialize member variables from parameters
    this->rimin_ = rimin;
    this->gimin_ = gimin;
    this->bimin_ = bimin;
    this->aimin_ = aimin;
    this->rimax_ = rimax;
    this->gimax_ = gimax;
    this->bimax_ = bimax;
    this->aimax_ = aimax;
    this->romin_ = romin;
    this->gomin_ = gomin;
    this->bomin_ = bomin;
    this->aomin_ = aomin;
    this->romax_ = romax;
    this->gomax_ = gomax;
    this->bomax_ = bomax;
    this->aomax_ = aomax;
    this->preserve_ = preserve;
}

Colorlevels::~Colorlevels() {
    // Destructor implementation (if needed)
}

void Colorlevels::setRimin(double value) {
    rimin_ = value;
}

double Colorlevels::getRimin() const {
    return rimin_;
}

void Colorlevels::setGimin(double value) {
    gimin_ = value;
}

double Colorlevels::getGimin() const {
    return gimin_;
}

void Colorlevels::setBimin(double value) {
    bimin_ = value;
}

double Colorlevels::getBimin() const {
    return bimin_;
}

void Colorlevels::setAimin(double value) {
    aimin_ = value;
}

double Colorlevels::getAimin() const {
    return aimin_;
}

void Colorlevels::setRimax(double value) {
    rimax_ = value;
}

double Colorlevels::getRimax() const {
    return rimax_;
}

void Colorlevels::setGimax(double value) {
    gimax_ = value;
}

double Colorlevels::getGimax() const {
    return gimax_;
}

void Colorlevels::setBimax(double value) {
    bimax_ = value;
}

double Colorlevels::getBimax() const {
    return bimax_;
}

void Colorlevels::setAimax(double value) {
    aimax_ = value;
}

double Colorlevels::getAimax() const {
    return aimax_;
}

void Colorlevels::setRomin(double value) {
    romin_ = value;
}

double Colorlevels::getRomin() const {
    return romin_;
}

void Colorlevels::setGomin(double value) {
    gomin_ = value;
}

double Colorlevels::getGomin() const {
    return gomin_;
}

void Colorlevels::setBomin(double value) {
    bomin_ = value;
}

double Colorlevels::getBomin() const {
    return bomin_;
}

void Colorlevels::setAomin(double value) {
    aomin_ = value;
}

double Colorlevels::getAomin() const {
    return aomin_;
}

void Colorlevels::setRomax(double value) {
    romax_ = value;
}

double Colorlevels::getRomax() const {
    return romax_;
}

void Colorlevels::setGomax(double value) {
    gomax_ = value;
}

double Colorlevels::getGomax() const {
    return gomax_;
}

void Colorlevels::setBomax(double value) {
    bomax_ = value;
}

double Colorlevels::getBomax() const {
    return bomax_;
}

void Colorlevels::setAomax(double value) {
    aomax_ = value;
}

double Colorlevels::getAomax() const {
    return aomax_;
}

void Colorlevels::setPreserve(int value) {
    preserve_ = value;
}

int Colorlevels::getPreserve() const {
    return preserve_;
}

std::string Colorlevels::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorlevels";

    bool first = true;

    if (rimin_ != 0.00) {
        desc << (first ? "=" : ":") << "rimin=" << rimin_;
        first = false;
    }
    if (gimin_ != 0.00) {
        desc << (first ? "=" : ":") << "gimin=" << gimin_;
        first = false;
    }
    if (bimin_ != 0.00) {
        desc << (first ? "=" : ":") << "bimin=" << bimin_;
        first = false;
    }
    if (aimin_ != 0.00) {
        desc << (first ? "=" : ":") << "aimin=" << aimin_;
        first = false;
    }
    if (rimax_ != 1.00) {
        desc << (first ? "=" : ":") << "rimax=" << rimax_;
        first = false;
    }
    if (gimax_ != 1.00) {
        desc << (first ? "=" : ":") << "gimax=" << gimax_;
        first = false;
    }
    if (bimax_ != 1.00) {
        desc << (first ? "=" : ":") << "bimax=" << bimax_;
        first = false;
    }
    if (aimax_ != 1.00) {
        desc << (first ? "=" : ":") << "aimax=" << aimax_;
        first = false;
    }
    if (romin_ != 0.00) {
        desc << (first ? "=" : ":") << "romin=" << romin_;
        first = false;
    }
    if (gomin_ != 0.00) {
        desc << (first ? "=" : ":") << "gomin=" << gomin_;
        first = false;
    }
    if (bomin_ != 0.00) {
        desc << (first ? "=" : ":") << "bomin=" << bomin_;
        first = false;
    }
    if (aomin_ != 0.00) {
        desc << (first ? "=" : ":") << "aomin=" << aomin_;
        first = false;
    }
    if (romax_ != 1.00) {
        desc << (first ? "=" : ":") << "romax=" << romax_;
        first = false;
    }
    if (gomax_ != 1.00) {
        desc << (first ? "=" : ":") << "gomax=" << gomax_;
        first = false;
    }
    if (bomax_ != 1.00) {
        desc << (first ? "=" : ":") << "bomax=" << bomax_;
        first = false;
    }
    if (aomax_ != 1.00) {
        desc << (first ? "=" : ":") << "aomax=" << aomax_;
        first = false;
    }
    if (preserve_ != 0) {
        desc << (first ? "=" : ":") << "preserve=" << preserve_;
        first = false;
    }

    return desc.str();
}
