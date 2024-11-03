#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Monochrome : public FilterBase {
public:
    /**
     * Convert video to gray using custom color filter.
     */
    /**
     * set the chroma blue spot
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setCb(float value);
    float getCb() const;

    /**
     * set the chroma red spot
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setCr(float value);
    float getCr() const;

    /**
     * set the color filter size
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setSize(float value);
    float getSize() const;

    /**
     * set the highlights strength
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setHigh(float value);
    float getHigh() const;

    Monochrome(float cb = 0.00, float cr = 0.00, float size = 1.00, float high = 0.00);
    virtual ~Monochrome();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float cb_;
    float cr_;
    float size_;
    float high_;
};
