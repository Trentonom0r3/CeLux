#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Entropy : public FilterBase {
public:
    /**
     * Measure video frames entropy.
     */
    /**
     * set kind of histogram entropy measurement
     * Unit: mode
     * Possible Values: normal (0), diff (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    Entropy(int mode = 0);
    virtual ~Entropy();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
};
