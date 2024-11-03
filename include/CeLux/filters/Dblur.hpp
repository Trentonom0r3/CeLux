#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Dblur : public FilterBase {
public:
    /**
     * Apply Directional Blur filter.
     */
    /**
     * set angle
     * Type: Float
     * Required: No
     * Default: 45.00
     */
    void setAngle(float value);
    float getAngle() const;

    /**
     * set radius
     * Type: Float
     * Required: No
     * Default: 5.00
     */
    void setRadius(float value);
    float getRadius() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Dblur(float angle = 45.00, float radius = 5.00, int planes = 15);
    virtual ~Dblur();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float angle_;
    float radius_;
    int planes_;
};
