#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Tonemap : public FilterBase {
public:
    /**
     * Conversion to/from different dynamic ranges.
     */
    /**
     * tonemap algorithm selection
     * Unit: tonemap
     * Possible Values: none (0), linear (1), gamma (2), clip (3), reinhard (4), hable (5), mobius (6)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setTonemap(int value);
    int getTonemap() const;

    /**
     * tonemap parameter
     * Type: Double
     * Required: No
     * Default: nan
     */
    void setParam(double value);
    double getParam() const;

    /**
     * desaturation strength
     * Type: Double
     * Required: No
     * Default: 2.00
     */
    void setDesat(double value);
    double getDesat() const;

    /**
     * signal peak override
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setPeak(double value);
    double getPeak() const;

    Tonemap(int tonemap = 0, double param = 0, double desat = 2.00, double peak = 0.00);
    virtual ~Tonemap();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int tonemap_;
    double param_;
    double desat_;
    double peak_;
};
