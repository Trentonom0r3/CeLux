#include "Showcqt.hpp"
#include <sstream>

Showcqt::Showcqt(std::pair<int, int> size, std::pair<int, int> rate, int bar_h, int axis_h, int sono_h, bool fullhd, const std::string& volume, const std::string& volume2, float sono_g, float gamma2, float bar_t, double timeclamp, double attack, double basefreq, double endfreq, float coeffclamp, const std::string& tlength, int count, int fcount, const std::string& fontfile, const std::string& font, const std::string& fontcolor, const std::string& axisfile, bool text, int csp, const std::string& cscheme) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->bar_h_ = bar_h;
    this->axis_h_ = axis_h;
    this->sono_h_ = sono_h;
    this->fullhd_ = fullhd;
    this->volume_ = volume;
    this->volume2_ = volume2;
    this->sono_g_ = sono_g;
    this->gamma2_ = gamma2;
    this->bar_t_ = bar_t;
    this->timeclamp_ = timeclamp;
    this->attack_ = attack;
    this->basefreq_ = basefreq;
    this->endfreq_ = endfreq;
    this->coeffclamp_ = coeffclamp;
    this->tlength_ = tlength;
    this->count_ = count;
    this->fcount_ = fcount;
    this->fontfile_ = fontfile;
    this->font_ = font;
    this->fontcolor_ = fontcolor;
    this->axisfile_ = axisfile;
    this->text_ = text;
    this->csp_ = csp;
    this->cscheme_ = cscheme;
}

Showcqt::~Showcqt() {
    // Destructor implementation (if needed)
}

void Showcqt::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Showcqt::getSize() const {
    return size_;
}

void Showcqt::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Showcqt::getRate() const {
    return rate_;
}

void Showcqt::setBar_h(int value) {
    bar_h_ = value;
}

int Showcqt::getBar_h() const {
    return bar_h_;
}

void Showcqt::setAxis_h(int value) {
    axis_h_ = value;
}

int Showcqt::getAxis_h() const {
    return axis_h_;
}

void Showcqt::setSono_h(int value) {
    sono_h_ = value;
}

int Showcqt::getSono_h() const {
    return sono_h_;
}

void Showcqt::setFullhd(bool value) {
    fullhd_ = value;
}

bool Showcqt::getFullhd() const {
    return fullhd_;
}

void Showcqt::setVolume(const std::string& value) {
    volume_ = value;
}

std::string Showcqt::getVolume() const {
    return volume_;
}

void Showcqt::setVolume2(const std::string& value) {
    volume2_ = value;
}

std::string Showcqt::getVolume2() const {
    return volume2_;
}

void Showcqt::setSono_g(float value) {
    sono_g_ = value;
}

float Showcqt::getSono_g() const {
    return sono_g_;
}

void Showcqt::setGamma2(float value) {
    gamma2_ = value;
}

float Showcqt::getGamma2() const {
    return gamma2_;
}

void Showcqt::setBar_t(float value) {
    bar_t_ = value;
}

float Showcqt::getBar_t() const {
    return bar_t_;
}

void Showcqt::setTimeclamp(double value) {
    timeclamp_ = value;
}

double Showcqt::getTimeclamp() const {
    return timeclamp_;
}

void Showcqt::setAttack(double value) {
    attack_ = value;
}

double Showcqt::getAttack() const {
    return attack_;
}

void Showcqt::setBasefreq(double value) {
    basefreq_ = value;
}

double Showcqt::getBasefreq() const {
    return basefreq_;
}

void Showcqt::setEndfreq(double value) {
    endfreq_ = value;
}

double Showcqt::getEndfreq() const {
    return endfreq_;
}

void Showcqt::setCoeffclamp(float value) {
    coeffclamp_ = value;
}

float Showcqt::getCoeffclamp() const {
    return coeffclamp_;
}

void Showcqt::setTlength(const std::string& value) {
    tlength_ = value;
}

std::string Showcqt::getTlength() const {
    return tlength_;
}

void Showcqt::setCount(int value) {
    count_ = value;
}

int Showcqt::getCount() const {
    return count_;
}

void Showcqt::setFcount(int value) {
    fcount_ = value;
}

int Showcqt::getFcount() const {
    return fcount_;
}

void Showcqt::setFontfile(const std::string& value) {
    fontfile_ = value;
}

std::string Showcqt::getFontfile() const {
    return fontfile_;
}

void Showcqt::setFont(const std::string& value) {
    font_ = value;
}

std::string Showcqt::getFont() const {
    return font_;
}

void Showcqt::setFontcolor(const std::string& value) {
    fontcolor_ = value;
}

std::string Showcqt::getFontcolor() const {
    return fontcolor_;
}

void Showcqt::setAxisfile(const std::string& value) {
    axisfile_ = value;
}

std::string Showcqt::getAxisfile() const {
    return axisfile_;
}

void Showcqt::setText(bool value) {
    text_ = value;
}

bool Showcqt::getText() const {
    return text_;
}

void Showcqt::setCsp(int value) {
    csp_ = value;
}

int Showcqt::getCsp() const {
    return csp_;
}

void Showcqt::setCscheme(const std::string& value) {
    cscheme_ = value;
}

std::string Showcqt::getCscheme() const {
    return cscheme_;
}

std::string Showcqt::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showcqt";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (bar_h_ != -1) {
        desc << (first ? "=" : ":") << "bar_h=" << bar_h_;
        first = false;
    }
    if (axis_h_ != -1) {
        desc << (first ? "=" : ":") << "axis_h=" << axis_h_;
        first = false;
    }
    if (sono_h_ != -1) {
        desc << (first ? "=" : ":") << "sono_h=" << sono_h_;
        first = false;
    }
    if (fullhd_ != true) {
        desc << (first ? "=" : ":") << "fullhd=" << (fullhd_ ? "1" : "0");
        first = false;
    }
    if (volume_ != "16") {
        desc << (first ? "=" : ":") << "volume=" << volume_;
        first = false;
    }
    if (volume2_ != "sono_v") {
        desc << (first ? "=" : ":") << "volume2=" << volume2_;
        first = false;
    }
    if (sono_g_ != 3.00) {
        desc << (first ? "=" : ":") << "sono_g=" << sono_g_;
        first = false;
    }
    if (gamma2_ != 1.00) {
        desc << (first ? "=" : ":") << "gamma2=" << gamma2_;
        first = false;
    }
    if (bar_t_ != 1.00) {
        desc << (first ? "=" : ":") << "bar_t=" << bar_t_;
        first = false;
    }
    if (timeclamp_ != 0.17) {
        desc << (first ? "=" : ":") << "timeclamp=" << timeclamp_;
        first = false;
    }
    if (attack_ != 0.00) {
        desc << (first ? "=" : ":") << "attack=" << attack_;
        first = false;
    }
    if (basefreq_ != 20.02) {
        desc << (first ? "=" : ":") << "basefreq=" << basefreq_;
        first = false;
    }
    if (endfreq_ != 20495.60) {
        desc << (first ? "=" : ":") << "endfreq=" << endfreq_;
        first = false;
    }
    if (coeffclamp_ != 1.00) {
        desc << (first ? "=" : ":") << "coeffclamp=" << coeffclamp_;
        first = false;
    }
    if (tlength_ != "384*tc/(384+tc*f)") {
        desc << (first ? "=" : ":") << "tlength=" << tlength_;
        first = false;
    }
    if (count_ != 6) {
        desc << (first ? "=" : ":") << "count=" << count_;
        first = false;
    }
    if (fcount_ != 0) {
        desc << (first ? "=" : ":") << "fcount=" << fcount_;
        first = false;
    }
    if (!fontfile_.empty()) {
        desc << (first ? "=" : ":") << "fontfile=" << fontfile_;
        first = false;
    }
    if (!font_.empty()) {
        desc << (first ? "=" : ":") << "font=" << font_;
        first = false;
    }
    if (fontcolor_ != "st(0, (midi(f)-59.5)/12);st(1, if(between(ld(0),0,1), 0.5-0.5*cos(2*PI*ld(0)), 0));r(1-ld(1)) + b(ld(1))") {
        desc << (first ? "=" : ":") << "fontcolor=" << fontcolor_;
        first = false;
    }
    if (!axisfile_.empty()) {
        desc << (first ? "=" : ":") << "axisfile=" << axisfile_;
        first = false;
    }
    if (text_ != true) {
        desc << (first ? "=" : ":") << "text=" << (text_ ? "1" : "0");
        first = false;
    }
    if (csp_ != 2) {
        desc << (first ? "=" : ":") << "csp=" << csp_;
        first = false;
    }
    if (cscheme_ != "1|0.5|0|0|0.5|1") {
        desc << (first ? "=" : ":") << "cscheme=" << cscheme_;
        first = false;
    }

    return desc.str();
}
