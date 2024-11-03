#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Derain : public FilterBase {
public:
    /**
     * Apply derain filter to the input.
     */
    /**
     * filter type(derain/dehaze)
     * Unit: type
     * Possible Values: derain (0), dehaze (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFilter_type(int value);
    int getFilter_type() const;

    /**
     * DNN backend
     * Unit: backend
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setDnn_backend(int value);
    int getDnn_backend() const;

    /**
     * path to model file
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

    Derain(int filter_type = 0, int dnn_backend = 1, const std::string& model = "", const std::string& input = "x", const std::string& output = "y");
    virtual ~Derain();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int filter_type_;
    int dnn_backend_;
    std::string model_;
    std::string input_;
    std::string output_;
};
