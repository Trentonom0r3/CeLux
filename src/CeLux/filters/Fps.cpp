#include "Fps.hpp"
#include <sstream>

Fps::Fps(const std::string& fps, double start_time, int round, int eof_action) {
    // Initialize member variables from parameters
    this->fps_ = fps;
    this->start_time_ = start_time;
    this->round_ = round;
    this->eof_action_ = eof_action;
}

Fps::~Fps() {
    // Destructor implementation (if needed)
}

void Fps::setFps(const std::string& value) {
    fps_ = value;
}

std::string Fps::getFps() const {
    return fps_;
}

void Fps::setStart_time(double value) {
    start_time_ = value;
}

double Fps::getStart_time() const {
    return start_time_;
}

void Fps::setRound(int value) {
    round_ = value;
}

int Fps::getRound() const {
    return round_;
}

void Fps::setEof_action(int value) {
    eof_action_ = value;
}

int Fps::getEof_action() const {
    return eof_action_;
}

std::string Fps::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "fps";

    bool first = true;

    if (fps_ != "25") {
        desc << (first ? "=" : ":") << "fps=" << fps_;
        first = false;
    }
    if (start_time_ != 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00) {
        desc << (first ? "=" : ":") << "start_time=" << start_time_;
        first = false;
    }
    if (round_ != 5) {
        desc << (first ? "=" : ":") << "round=" << round_;
        first = false;
    }
    if (eof_action_ != 0) {
        desc << (first ? "=" : ":") << "eof_action=" << eof_action_;
        first = false;
    }

    return desc.str();
}
