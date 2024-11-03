#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Varblur : public FilterBase {
public:
    /**
     * Apply Variable Blur filter.
     */
    /**
     * set min blur radius
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMin_r(int value);
    int getMin_r() const;

    /**
     * set max blur radius
     * Type: Integer
     * Required: No
     * Default: 8
     */
    void setMax_r(int value);
    int getMax_r() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Varblur(int min_r = 0, int max_r = 8, int planes = 15);
    virtual ~Varblur();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int min_r_;
    int max_r_;
    int planes_;
};
