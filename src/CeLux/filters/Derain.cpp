#include "Derain.hpp"
#include <sstream>

Derain::Derain(int filter_type, int dnn_backend, const std::string& model, const std::string& input, const std::string& output) {
    // Initialize member variables from parameters
    this->filter_type_ = filter_type;
    this->dnn_backend_ = dnn_backend;
    this->model_ = model;
    this->input_ = input;
    this->output_ = output;
}

Derain::~Derain() {
    // Destructor implementation (if needed)
}

void Derain::setFilter_type(int value) {
    filter_type_ = value;
}

int Derain::getFilter_type() const {
    return filter_type_;
}

void Derain::setDnn_backend(int value) {
    dnn_backend_ = value;
}

int Derain::getDnn_backend() const {
    return dnn_backend_;
}

void Derain::setModel(const std::string& value) {
    model_ = value;
}

std::string Derain::getModel() const {
    return model_;
}

void Derain::setInput(const std::string& value) {
    input_ = value;
}

std::string Derain::getInput() const {
    return input_;
}

void Derain::setOutput(const std::string& value) {
    output_ = value;
}

std::string Derain::getOutput() const {
    return output_;
}

std::string Derain::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "derain";

    bool first = true;

    if (filter_type_ != 0) {
        desc << (first ? "=" : ":") << "filter_type=" << filter_type_;
        first = false;
    }
    if (dnn_backend_ != 1) {
        desc << (first ? "=" : ":") << "dnn_backend=" << dnn_backend_;
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
