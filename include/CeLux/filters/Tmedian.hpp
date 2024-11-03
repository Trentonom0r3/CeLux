#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Tmedian : public FilterBase {
public:
    /**
     * Pick median pixels from successive frames.
     */
    /**
     * set median filter radius
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
     * set percentile
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setPercentile(float value);
    float getPercentile() const;

    Tmedian(int radius = 1, int planes = 15, float percentile = 0.50);
    virtual ~Tmedian();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int radius_;
    int planes_;
    float percentile_;
};
