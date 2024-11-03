#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Lumakey : public FilterBase {
public:
    /**
     * Turns a certain luma into transparency.
     */
    /**
     * set the threshold value
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setThreshold(double value);
    double getThreshold() const;

    /**
     * set the tolerance value
     * Type: Double
     * Required: No
     * Default: 0.01
     */
    void setTolerance(double value);
    double getTolerance() const;

    /**
     * set the softness value
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setSoftness(double value);
    double getSoftness() const;

    Lumakey(double threshold = 0.00, double tolerance = 0.01, double softness = 0.00);
    virtual ~Lumakey();

    std::string getFilterDescription() const override;

private:
    // Option variables
    double threshold_;
    double tolerance_;
    double softness_;
};
