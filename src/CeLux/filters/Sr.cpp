#include "Sr.hpp"
#include <sstream>

Sr::Sr(int dnn_backend, int scale_factor, const std::string& model, const std::string& input, const std::string& output) {
    // Initialize member variables from parameters
    this->dnn_backend_ = dnn_backend;
    this->scale_factor_ = scale_factor;
    this->model_ = model;
    this->input_ = input;
    this->output_ = output;
}

Sr::~Sr() {
    // Destructor implementation (if needed)
}

void Sr::setDnn_backend(int value) {
    dnn_backend_ = value;
}

int Sr::getDnn_backend() const {
    return dnn_backend_;
}

void Sr::setScale_factor(int value) {
    scale_factor_ = value;
}

int Sr::getScale_factor() const {
    return scale_factor_;
}

void Sr::setModel(const std::string& value) {
    model_ = value;
}

std::string Sr::getModel() const {
    return model_;
}

void Sr::setInput(const std::string& value) {
    input_ = value;
}

std::string Sr::getInput() const {
    return input_;
}

void Sr::setOutput(const std::string& value) {
    output_ = value;
}

std::string Sr::getOutput() const {
    return output_;
}

std::string Sr::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "sr";

    bool first = true;

    if (dnn_backend_ != 1) {
        desc << (first ? "=" : ":") << "dnn_backend=" << dnn_backend_;
        first = false;
    }
    if (scale_factor_ != 2) {
        desc << (first ? "=" : ":") << "scale_factor=" << scale_factor_;
        first = false;
    }
    if (!model_.empty()) {
        desc << (first ? "=" : ":") << "model=" << model_;
        first = false;
    }
    if (input_ != "x") {
        desc << (first ? "=" : ":") << "input=" << input_;
        first = false;
    }
    if (output_ != "y") {
        desc << (first ? "=" : ":") << "output=" << output_;
        first = false;
    }

    return desc.str();
}
