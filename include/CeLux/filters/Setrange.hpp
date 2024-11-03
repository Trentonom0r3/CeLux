#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Setrange : public FilterBase {
public:
    /**
     * Force color range for the output video frame.
     */
    /**
     * select color range
     * Unit: range
     * Possible Values: auto (-1), unspecified (0), unknown (0), limited (1), tv (1), mpeg (1), full (2), pc (2), jpeg (2)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setRange(int value);
    int getRange() const;

    Setrange(int range = -1);
    virtual ~Setrange();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int range_;
};
