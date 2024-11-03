#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Multiply : public FilterBase {
public:
    /**
     * Multiply first video stream with second video stream.
     */
    /**
     * set scale
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setScale(float value);
    float getScale() const;

    /**
     * set offset
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setOffset(float value);
    float getOffset() const;

    /**
     * set planes
     * Type: Flags
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Multiply(float scale = 1.00, float offset = 0.50, int planes = 15);
    virtual ~Multiply();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float scale_;
    float offset_;
    int planes_;
};
