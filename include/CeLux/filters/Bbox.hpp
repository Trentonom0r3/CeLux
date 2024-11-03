#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Bbox : public FilterBase {
public:
    /**
     * Compute bounding box for each frame.
     */
    /**
     * set minimum luminance value for bounding box
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setMin_val(int value);
    int getMin_val() const;

    Bbox(int min_val = 16);
    virtual ~Bbox();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int min_val_;
};
