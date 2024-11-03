#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Select : public FilterBase {
public:
    /**
     * Select video frames to pass in output.
     */
    /**
     * set an expression to use for selecting frames
     * Aliases: e
     * Type: String
     * Required: No
     * Default: 1
     */
    void setExpr(const std::string& value);
    std::string getExpr() const;

    /**
     * set the number of outputs
     * Aliases: n
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setOutputs(int value);
    int getOutputs() const;

    Select(const std::string& expr = "1", int outputs = 1);
    virtual ~Select();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string expr_;
    int outputs_;
};
