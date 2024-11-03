#include "Showinfo.hpp"
#include <sstream>

Showinfo::Showinfo(bool checksum, bool udu_sei_as_ascii) {
    // Initialize member variables from parameters
    this->checksum_ = checksum;
    this->udu_sei_as_ascii_ = udu_sei_as_ascii;
}

Showinfo::~Showinfo() {
    // Destructor implementation (if needed)
}

void Showinfo::setChecksum(bool value) {
    checksum_ = value;
}

bool Showinfo::getChecksum() const {
    return checksum_;
}

void Showinfo::setUdu_sei_as_ascii(bool value) {
    udu_sei_as_ascii_ = value;
}

bool Showinfo::getUdu_sei_as_ascii() const {
    return udu_sei_as_ascii_;
}

std::string Showinfo::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showinfo";

    bool first = true;

    if (checksum_ != true) {
        desc << (first ? "=" : ":") << "checksum=" << (checksum_ ? "1" : "0");
        first = false;
    }
    if (udu_sei_as_ascii_ != false) {
        desc << (first ? "=" : ":") << "udu_sei_as_ascii=" << (udu_sei_as_ascii_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
