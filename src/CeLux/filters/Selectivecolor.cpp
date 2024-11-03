#include "Selectivecolor.hpp"
#include <sstream>

Selectivecolor::Selectivecolor(int correction_method, const std::string& reds, const std::string& yellows, const std::string& greens, const std::string& cyans, const std::string& blues, const std::string& magentas, const std::string& whites, const std::string& neutrals, const std::string& blacks, const std::string& psfile) {
    // Initialize member variables from parameters
    this->correction_method_ = correction_method;
    this->reds_ = reds;
    this->yellows_ = yellows;
    this->greens_ = greens;
    this->cyans_ = cyans;
    this->blues_ = blues;
    this->magentas_ = magentas;
    this->whites_ = whites;
    this->neutrals_ = neutrals;
    this->blacks_ = blacks;
    this->psfile_ = psfile;
}

Selectivecolor::~Selectivecolor() {
    // Destructor implementation (if needed)
}

void Selectivecolor::setCorrection_method(int value) {
    correction_method_ = value;
}

int Selectivecolor::getCorrection_method() const {
    return correction_method_;
}

void Selectivecolor::setReds(const std::string& value) {
    reds_ = value;
}

std::string Selectivecolor::getReds() const {
    return reds_;
}

void Selectivecolor::setYellows(const std::string& value) {
    yellows_ = value;
}

std::string Selectivecolor::getYellows() const {
    return yellows_;
}

void Selectivecolor::setGreens(const std::string& value) {
    greens_ = value;
}

std::string Selectivecolor::getGreens() const {
    return greens_;
}

void Selectivecolor::setCyans(const std::string& value) {
    cyans_ = value;
}

std::string Selectivecolor::getCyans() const {
    return cyans_;
}

void Selectivecolor::setBlues(const std::string& value) {
    blues_ = value;
}

std::string Selectivecolor::getBlues() const {
    return blues_;
}

void Selectivecolor::setMagentas(const std::string& value) {
    magentas_ = value;
}

std::string Selectivecolor::getMagentas() const {
    return magentas_;
}

void Selectivecolor::setWhites(const std::string& value) {
    whites_ = value;
}

std::string Selectivecolor::getWhites() const {
    return whites_;
}

void Selectivecolor::setNeutrals(const std::string& value) {
    neutrals_ = value;
}

std::string Selectivecolor::getNeutrals() const {
    return neutrals_;
}

void Selectivecolor::setBlacks(const std::string& value) {
    blacks_ = value;
}

std::string Selectivecolor::getBlacks() const {
    return blacks_;
}

void Selectivecolor::setPsfile(const std::string& value) {
    psfile_ = value;
}

std::string Selectivecolor::getPsfile() const {
    return psfile_;
}

std::string Selectivecolor::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "selectivecolor";

    bool first = true;

    if (correction_method_ != 0) {
        desc << (first ? "=" : ":") << "correction_method=" << correction_method_;
        first = false;
    }
    if (!reds_.empty()) {
        desc << (first ? "=" : ":") << "reds=" << reds_;
        first = false;
    }
    if (!yellows_.empty()) {
        desc << (first ? "=" : ":") << "yellows=" << yellows_;
        first = false;
    }
    if (!greens_.empty()) {
        desc << (first ? "=" : ":") << "greens=" << greens_;
        first = false;
    }
    if (!cyans_.empty()) {
        desc << (first ? "=" : ":") << "cyans=" << cyans_;
        first = false;
    }
    if (!blues_.empty()) {
        desc << (first ? "=" : ":") << "blues=" << blues_;
        first = false;
    }
    if (!magentas_.empty()) {
        desc << (first ? "=" : ":") << "magentas=" << magentas_;
        first = false;
    }
    if (!whites_.empty()) {
        desc << (first ? "=" : ":") << "whites=" << whites_;
        first = false;
    }
    if (!neutrals_.empty()) {
        desc << (first ? "=" : ":") << "neutrals=" << neutrals_;
        first = false;
    }
    if (!blacks_.empty()) {
        desc << (first ? "=" : ":") << "blacks=" << blacks_;
        first = false;
    }
    if (!psfile_.empty()) {
        desc << (first ? "=" : ":") << "psfile=" << psfile_;
        first = false;
    }

    return desc.str();
}
