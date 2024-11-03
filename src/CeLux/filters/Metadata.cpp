#include "Metadata.hpp"
#include <sstream>

Metadata::Metadata(int mode, const std::string& key, const std::string& value, int function, const std::string& expr, const std::string& file, bool direct) {
    // Initialize member variables from parameters
    this->mode_ = mode;
    this->key_ = key;
    this->value_ = value;
    this->function_ = function;
    this->expr_ = expr;
    this->file_ = file;
    this->direct_ = direct;
}

Metadata::~Metadata() {
    // Destructor implementation (if needed)
}

void Metadata::setMode(int value) {
    mode_ = value;
}

int Metadata::getMode() const {
    return mode_;
}

void Metadata::setKey(const std::string& value) {
    key_ = value;
}

std::string Metadata::getKey() const {
    return key_;
}

void Metadata::setValue(const std::string& value) {
    value_ = value;
}

std::string Metadata::getValue() const {
    return value_;
}

void Metadata::setFunction(int value) {
    function_ = value;
}

int Metadata::getFunction() const {
    return function_;
}

void Metadata::setExpr(const std::string& value) {
    expr_ = value;
}

std::string Metadata::getExpr() const {
    return expr_;
}

void Metadata::setFile(const std::string& value) {
    file_ = value;
}

std::string Metadata::getFile() const {
    return file_;
}

void Metadata::setDirect(bool value) {
    direct_ = value;
}

bool Metadata::getDirect() const {
    return direct_;
}

std::string Metadata::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "metadata";

    bool first = true;

    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (!key_.empty()) {
        desc << (first ? "=" : ":") << "key=" << key_;
        first = false;
    }
    if (!value_.empty()) {
        desc << (first ? "=" : ":") << "value=" << value_;
        first = false;
    }
    if (function_ != 0) {
        desc << (first ? "=" : ":") << "function=" << function_;
        first = false;
    }
    if (!expr_.empty()) {
        desc << (first ? "=" : ":") << "expr=" << expr_;
        first = false;
    }
    if (!file_.empty()) {
        desc << (first ? "=" : ":") << "file=" << file_;
        first = false;
    }
    if (direct_ != false) {
        desc << (first ? "=" : ":") << "direct=" << (direct_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
