#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Lenscorrection : public FilterBase {
public:
    /**
     * Rectify the image by correcting for lens distortion.
     */
    /**
     * set relative center x
     * Type: Double
     * Required: No
     * Default: 0.50
     */
    void setCx(double value);
    double getCx() const;

    /**
     * set relative center y
     * Type: Double
     * Required: No
     * Default: 0.50
     */
    void setCy(double value);
    double getCy() const;

    /**
     * set quadratic distortion factor
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setK1(double value);
    double getK1() const;

    /**
     * set double quadratic distortion factor
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setK2(double value);
    double getK2() const;

    /**
     * set interpolation type
     * Unit: i
     * Possible Values: nearest (0), bilinear (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setInterpolationType(int value);
    int getInterpolationType() const;

    /**
     * set the color of the unmapped pixels
     * Type: Color
     * Required: No
     * Default: black@0
     */
    void setFc(const std::string& value);
    std::string getFc() const;

    Lenscorrection(double cx = 0.50, double cy = 0.50, double k1 = 0.00, double k2 = 0.00, int interpolationType = 0, const std::string& fc = "black@0");
    virtual ~Lenscorrection();

    std::string getFilterDescription() const override;

private:
    // Option variables
    double cx_;
    double cy_;
    double k1_;
    double k2_;
    int interpolationType_;
    std::string fc_;
};
