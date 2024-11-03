#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Lut1d : public FilterBase {
public:
    /**
     * Adjust colors using a 1D LUT.
     */
    /**
     * set 1D LUT file name
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFile(const std::string& value);
    std::string getFile() const;

    /**
     * select interpolation mode
     * Unit: interp_mode
     * Possible Values: nearest (0), linear (1), cosine (3), cubic (2), spline (4)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setInterp(int value);
    int getInterp() const;

    Lut1d(const std::string& file = "", int interp = 1);
    virtual ~Lut1d();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string file_;
    int interp_;
};
