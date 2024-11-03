#include "Cellauto.hpp"
#include <sstream>

Cellauto::Cellauto(const std::string& filename, const std::string& pattern, std::pair<int, int> rate, std::pair<int, int> size, int rule, double random_fill_ratio, int64_t random_seed, bool scroll, bool start_full, bool full, bool stitch) {
    // Initialize member variables from parameters
    this->filename_ = filename;
    this->pattern_ = pattern;
    this->rate_ = rate;
    this->size_ = size;
    this->rule_ = rule;
    this->random_fill_ratio_ = random_fill_ratio;
    this->random_seed_ = random_seed;
    this->scroll_ = scroll;
    this->start_full_ = start_full;
    this->full_ = full;
    this->stitch_ = stitch;
}

Cellauto::~Cellauto() {
    // Destructor implementation (if needed)
}

void Cellauto::setFilename(const std::string& value) {
    filename_ = value;
}

std::string Cellauto::getFilename() const {
    return filename_;
}

void Cellauto::setPattern(const std::string& value) {
    pattern_ = value;
}

std::string Cellauto::getPattern() const {
    return pattern_;
}

void Cellauto::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Cellauto::getRate() const {
    return rate_;
}

void Cellauto::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Cellauto::getSize() const {
    return size_;
}

void Cellauto::setRule(int value) {
    rule_ = value;
}

int Cellauto::getRule() const {
    return rule_;
}

void Cellauto::setRandom_fill_ratio(double value) {
    random_fill_ratio_ = value;
}

double Cellauto::getRandom_fill_ratio() const {
    return random_fill_ratio_;
}

void Cellauto::setRandom_seed(int64_t value) {
    random_seed_ = value;
}

int64_t Cellauto::getRandom_seed() const {
    return random_seed_;
}

void Cellauto::setScroll(bool value) {
    scroll_ = value;
}

bool Cellauto::getScroll() const {
    return scroll_;
}

void Cellauto::setStart_full(bool value) {
    start_full_ = value;
}

bool Cellauto::getStart_full() const {
    return start_full_;
}

void Cellauto::setFull(bool value) {
    full_ = value;
}

bool Cellauto::getFull() const {
    return full_;
}

void Cellauto::setStitch(bool value) {
    stitch_ = value;
}

bool Cellauto::getStitch() const {
    return stitch_;
}

std::string Cellauto::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "cellauto";

    bool first = true;

    if (!filename_.empty()) {
        desc << (first ? "=" : ":") << "filename=" << filename_;
        first = false;
    }
    if (!pattern_.empty()) {
        desc << (first ? "=" : ":") << "pattern=" << pattern_;
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
    if (rule_ != 110) {
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
    if (scroll_ != true) {
        desc << (first ? "=" : ":") << "scroll=" << (scroll_ ? "1" : "0");
        first = false;
    }
    if (start_full_ != false) {
        desc << (first ? "=" : ":") << "start_full=" << (start_full_ ? "1" : "0");
        first = false;
    }
    if (full_ != true) {
        desc << (first ? "=" : ":") << "full=" << (full_ ? "1" : "0");
        first = false;
    }
    if (stitch_ != true) {
        desc << (first ? "=" : ":") << "stitch=" << (stitch_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
