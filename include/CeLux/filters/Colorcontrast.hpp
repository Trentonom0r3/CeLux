#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorcontrast : public FilterBase {
public:
    /**
     * Adjust color contrast between RGB components.
     */
    /**
     * set the red-cyan contrast
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRc(float value);
    float getRc() const;

    /**
     * set the green-magenta contrast
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setGm(float value);
    float getGm() const;

    /**
     * set the blue-yellow contrast
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBy(float value);
    float getBy() const;

    /**
     * set the red-cyan weight
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRcw(float value);
    float getRcw() const;

    /**
     * set the green-magenta weight
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setGmw(float value);
    float getGmw() const;

    /**
     * set the blue-yellow weight
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setByw(float value);
    float getByw() const;

    /**
     * set the amount of preserving lightness
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setPl(float value);
    float getPl() const;

    Colorcontrast(float rc = 0.00, float gm = 0.00, float by = 0.00, float rcw = 0.00, float gmw = 0.00, float byw = 0.00, float pl = 0.00);
    virtual ~Colorcontrast();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float rc_;
    float gm_;
    float by_;
    float rcw_;
    float gmw_;
    float byw_;
    float pl_;
};
