#include "Life.hpp"
#include <sstream>

Life::Life(const std::string& filename, std::pair<int, int> size, std::pair<int, int> rate, const std::string& rule, double random_fill_ratio, int64_t random_seed, bool stitch, int mold, const std::string& life_color, const std::string& death_color, const std::string& mold_color) {
    // Initialize member variables from parameters
    this->filename_ = filename;
    this->size_ = size;
    this->rate_ = rate;
    this->rule_ = rule;
    this->random_fill_ratio_ = random_fill_ratio;
    this->random_seed_ = random_seed;
    this->stitch_ = stitch;
    this->mold_ = mold;
    this->life_color_ = life_color;
    this->death_color_ = death_color;
    this->mold_color_ = mold_color;
}

Life::~Life() {
    // Destructor implementation (if needed)
}

void Life::setFilename(const std::string& value) {
    filename_ = value;
}

std::string Life::getFilename() const {
    return filename_;
}

void Life::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Life::getSize() const {
    return size_;
}

void Life::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Life::getRate() const {
    return rate_;
}

void Life::setRule(const std::string& value) {
    rule_ = value;
}

std::string Life::getRule() const {
    return rule_;
}

void Life::setRandom_fill_ratio(double value) {
    random_fill_ratio_ = value;
}

double Life::getRandom_fill_ratio() const {
    return random_fill_ratio_;
}

void Life::setRandom_seed(int64_t value) {
    random_seed_ = value;
}

int64_t Life::getRandom_seed() const {
    return random_seed_;
}

void Life::setStitch(bool value) {
    stitch_ = value;
}

bool Life::getStitch() const {
    return stitch_;
}

void Life::setMold(int value) {
    mold_ = value;
}

int Life::getMold() const {
    return mold_;
}

void Life::setLife_color(const std::string& value) {
    life_color_ = value;
}

std::string Life::getLife_color() const {
    return life_color_;
}

void Life::setDeath_color(const std::string& value) {
    death_color_ = value;
}

std::string Life::getDeath_color() const {
    return death_color_;
}

void Life::setMold_color(const std::string& value) {
    mold_color_ = value;
}

std::string Life::getMold_color() const {
    return mold_color_;
}

std::string Life::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "life";

    bool first = true;

    if (!filename_.empty()) {
        desc << (first ? "=" : ":") << "filename=" << filename_;
        first = false;
    }
    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (rule_ != "B3/S23") {
        desc << (first ? "=" : ":") << "rule=" << rule_;
        first = false;
    }
    if (random_fill_ratio_ != 0.62) {
        desc << (first ? "=" : ":") << "random_fill_ratio=" << random_fill_ratio_;
        first = false;
    }
    if (random_seed_ != 0) {
        desc << (first ? "=" : ":") << "random_seed=" << random_seed_;
        first = false;
    }
    if (stitch_ != true) {
        desc << (first ? "=" : ":") << "stitch=" << (stitch_ ? "1" : "0");
        first = false;
    }
    if (mold_ != 0) {
        desc << (first ? "=" : ":") << "mold=" << mold_;
        first = false;
    }
    if (life_color_ != "white") {
        desc << (first ? "=" : ":") << "life_color=" << life_color_;
        first = false;
    }
    if (death_color_ != "black") {
        desc << (first ? "=" : ":") << "death_color=" << death_color_;
        first = false;
    }
    if (mold_color_ != "black") {
        desc << (first ? "=" : ":") << "mold_color=" << mold_color_;
        first = false;
    }

    return desc.str();
}
