#include "Showvolume.hpp"
#include <sstream>

Showvolume::Showvolume(std::pair<int, int> rate, int borderWidth, int channelWidth, int channelHeight, double fade, const std::string& volumeColor, bool displayChannelNames, bool displayVolume, double dm, const std::string& dmc, int orientation, int stepSize, float backgroundOpacity, int mode, int ds) {
    // Initialize member variables from parameters
    this->rate_ = rate;
    this->borderWidth_ = borderWidth;
    this->channelWidth_ = channelWidth;
    this->channelHeight_ = channelHeight;
    this->fade_ = fade;
    this->volumeColor_ = volumeColor;
    this->displayChannelNames_ = displayChannelNames;
    this->displayVolume_ = displayVolume;
    this->dm_ = dm;
    this->dmc_ = dmc;
    this->orientation_ = orientation;
    this->stepSize_ = stepSize;
    this->backgroundOpacity_ = backgroundOpacity;
    this->mode_ = mode;
    this->ds_ = ds;
}

Showvolume::~Showvolume() {
    // Destructor implementation (if needed)
}

void Showvolume::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Showvolume::getRate() const {
    return rate_;
}

void Showvolume::setBorderWidth(int value) {
    borderWidth_ = value;
}

int Showvolume::getBorderWidth() const {
    return borderWidth_;
}

void Showvolume::setChannelWidth(int value) {
    channelWidth_ = value;
}

int Showvolume::getChannelWidth() const {
    return channelWidth_;
}

void Showvolume::setChannelHeight(int value) {
    channelHeight_ = value;
}

int Showvolume::getChannelHeight() const {
    return channelHeight_;
}

void Showvolume::setFade(double value) {
    fade_ = value;
}

double Showvolume::getFade() const {
    return fade_;
}

void Showvolume::setVolumeColor(const std::string& value) {
    volumeColor_ = value;
}

std::string Showvolume::getVolumeColor() const {
    return volumeColor_;
}

void Showvolume::setDisplayChannelNames(bool value) {
    displayChannelNames_ = value;
}

bool Showvolume::getDisplayChannelNames() const {
    return displayChannelNames_;
}

void Showvolume::setDisplayVolume(bool value) {
    displayVolume_ = value;
}

bool Showvolume::getDisplayVolume() const {
    return displayVolume_;
}

void Showvolume::setDm(double value) {
    dm_ = value;
}

double Showvolume::getDm() const {
    return dm_;
}

void Showvolume::setDmc(const std::string& value) {
    dmc_ = value;
}

std::string Showvolume::getDmc() const {
    return dmc_;
}

void Showvolume::setOrientation(int value) {
    orientation_ = value;
}

int Showvolume::getOrientation() const {
    return orientation_;
}

void Showvolume::setStepSize(int value) {
    stepSize_ = value;
}

int Showvolume::getStepSize() const {
    return stepSize_;
}

void Showvolume::setBackgroundOpacity(float value) {
    backgroundOpacity_ = value;
}

float Showvolume::getBackgroundOpacity() const {
    return backgroundOpacity_;
}

void Showvolume::setMode(int value) {
    mode_ = value;
}

int Showvolume::getMode() const {
    return mode_;
}

void Showvolume::setDs(int value) {
    ds_ = value;
}

int Showvolume::getDs() const {
    return ds_;
}

std::string Showvolume::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showvolume";

    bool first = true;

    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (borderWidth_ != 1) {
        desc << (first ? "=" : ":") << "b=" << borderWidth_;
        first = false;
    }
    if (channelWidth_ != 400) {
        desc << (first ? "=" : ":") << "w=" << channelWidth_;
        first = false;
    }
    if (channelHeight_ != 20) {
        desc << (first ? "=" : ":") << "h=" << channelHeight_;
        first = false;
    }
    if (fade_ != 0.95) {
        desc << (first ? "=" : ":") << "f=" << fade_;
        first = false;
    }
    if (volumeColor_ != "PEAK*255+floor((1-PEAK)*255)*256+0xff000000") {
        desc << (first ? "=" : ":") << "c=" << volumeColor_;
        first = false;
    }
    if (displayChannelNames_ != true) {
        desc << (first ? "=" : ":") << "t=" << (displayChannelNames_ ? "1" : "0");
        first = false;
    }
    if (displayVolume_ != true) {
        desc << (first ? "=" : ":") << "v=" << (displayVolume_ ? "1" : "0");
        first = false;
    }
    if (dm_ != 0.00) {
        desc << (first ? "=" : ":") << "dm=" << dm_;
        first = false;
    }
    if (dmc_ != "orange") {
        desc << (first ? "=" : ":") << "dmc=" << dmc_;
        first = false;
    }
    if (orientation_ != 0) {
        desc << (first ? "=" : ":") << "o=" << orientation_;
        first = false;
    }
    if (stepSize_ != 0) {
        desc << (first ? "=" : ":") << "s=" << stepSize_;
        first = false;
    }
    if (backgroundOpacity_ != 0.00) {
        desc << (first ? "=" : ":") << "p=" << backgroundOpacity_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "m=" << mode_;
        first = false;
    }
    if (ds_ != 0) {
        desc << (first ? "=" : ":") << "ds=" << ds_;
        first = false;
    }

    return desc.str();
}
