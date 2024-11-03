#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Maskedthreshold : public FilterBase {
public:
    /**
     * Pick pixels comparing absolute difference of two streams with threshold.
     */
    /**
     * set threshold
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setThreshold(int value);
    int getThreshold() const;

    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * set mode
     * Unit: mode
     * Possible Values: abs (0), diff (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    Maskedthreshold(int threshold = 1, int planes = 15, int mode = 0);
    virtual ~Maskedthreshold();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int threshold_;
    int planes_;
    int mode_;
};
