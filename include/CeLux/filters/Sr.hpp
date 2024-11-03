#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Sr : public FilterBase {
public:
    /**
     * Apply DNN-based image super resolution to the input.
     */
    /**
     * DNN backend used for model execution
     * Unit: backend
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setDnn_backend(int value);
    int getDnn_backend() const;

    /**
     * scale factor for SRCNN model
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setScale_factor(int value);
    int getScale_factor() const;

    /**
     * path to model file specifying network architecture and its parameters
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setModel(const std::string& value);
    std::string getModel() const;

    /**
     * input name of the model
     * Type: String
     * Required: No
     * Default: x
     */
    void setInput(const std::string& value);
    std::string getInput() const;

    /**
     * output name of the model
     * Type: String
     * Required: No
     * Default: y
     */
    void setOutput(const std::string& value);
    std::string getOutput() const;

    Sr(int dnn_backend = 1, int scale_factor = 2, const std::string& model = "", const std::string& input = "x", const std::string& output = "y");
    virtual ~Sr();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int dnn_backend_;
    int scale_factor_;
    std::string model_;
    std::string input_;
    std::string output_;
};
