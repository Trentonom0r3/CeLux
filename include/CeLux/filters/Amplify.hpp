#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Amplify : public FilterBase {
public:
    /**
     * Amplify changes between successive video frames.
     */
    /**
     * set radius
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setRadius(int value);
    int getRadius() const;

    /**
     * set factor
     * Type: Float
     * Required: No
     * Default: 2.00
     */
    void setFactor(float value);
    float getFactor() const;

    /**
     * set threshold
     * Type: Float
     * Required: No
     * Default: 10.00
     */
    void setThreshold(float value);
    float getThreshold() const;

    /**
     * set tolerance
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setTolerance(float value);
    float getTolerance() const;

    /**
     * set low limit for amplification
     * Type: Float
     * Required: No
     * Default: 65535.00
     */
    void setLow(float value);
    float getLow() const;

    /**
     * set high limit for amplification
     * Type: Float
     * Required: No
     * Default: 65535.00
     */
    void setHigh(float value);
    float getHigh() const;

    /**
     * set what planes to filter
     * Type: Flags
     * Required: No
     * Default: 7
     */
    void setPlanes(int value);
    int getPlanes() const;

    Amplify(int radius = 2, float factor = 2.00, float threshold = 10.00, float tolerance = 0.00, float low = 65535.00, float high = 65535.00, int planes = 7);
    virtual ~Amplify();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int radius_;
    float factor_;
    float threshold_;
    float tolerance_;
    float low_;
    float high_;
    int planes_;
};
