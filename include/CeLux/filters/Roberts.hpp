#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Roberts : public FilterBase {
public:
    /**
     * Apply roberts cross operator.
     */
    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * set scale
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setScale(float value);
    float getScale() const;

    /**
     * set delta
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setDelta(float value);
    float getDelta() const;

    Roberts(int planes = 15, float scale = 1.00, float delta = 0.00);
    virtual ~Roberts();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
    float scale_;
    float delta_;
};
