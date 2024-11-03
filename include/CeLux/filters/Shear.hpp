#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Shear : public FilterBase {
public:
    /**
     * Shear transform the input image.
     */
    /**
     * set x shear factor
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setShx(float value);
    float getShx() const;

    /**
     * set y shear factor
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setShy(float value);
    float getShy() const;

    /**
     * set background fill color
     * Aliases: c
     * Type: String
     * Required: No
     * Default: black
     */
    void setFillcolor(const std::string& value);
    std::string getFillcolor() const;

    /**
     * set interpolation
     * Unit: interp
     * Possible Values: nearest (0), bilinear (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setInterp(int value);
    int getInterp() const;

    Shear(float shx = 0.00, float shy = 0.00, const std::string& fillcolor = "black", int interp = 1);
    virtual ~Shear();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float shx_;
    float shy_;
    std::string fillcolor_;
    int interp_;
};
