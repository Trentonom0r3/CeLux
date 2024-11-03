#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Median : public FilterBase {
public:
    /**
     * Apply Median filter.
     */
    /**
     * set median radius
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setRadius(int value);
    int getRadius() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * set median vertical radius
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setRadiusV(int value);
    int getRadiusV() const;

    /**
     * set median percentile
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setPercentile(float value);
    float getPercentile() const;

    Median(int radius = 1, int planes = 15, int radiusV = 0, float percentile = 0.50);
    virtual ~Median();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int radius_;
    int planes_;
    int radiusV_;
    float percentile_;
};
