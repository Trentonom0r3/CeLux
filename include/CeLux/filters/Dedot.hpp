#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Dedot : public FilterBase {
public:
    /**
     * Reduce cross-luminance and cross-color.
     */
    /**
     * set filtering mode
     * Unit: m
     * Possible Values: dotcrawl (1), rainbows (2)
     * Type: Flags
     * Required: No
     * Default: 3
     */
    void setFilteringMode(int value);
    int getFilteringMode() const;

    /**
     * set spatial luma threshold
     * Type: Float
     * Required: No
     * Default: 0.08
     */
    void setLt(float value);
    float getLt() const;

    /**
     * set tolerance for temporal luma
     * Type: Float
     * Required: No
     * Default: 0.08
     */
    void setTl(float value);
    float getTl() const;

    /**
     * set tolerance for chroma temporal variation
     * Type: Float
     * Required: No
     * Default: 0.06
     */
    void setTc(float value);
    float getTc() const;

    /**
     * set temporal chroma threshold
     * Type: Float
     * Required: No
     * Default: 0.02
     */
    void setCt(float value);
    float getCt() const;

    Dedot(int filteringMode = 3, float lt = 0.08, float tl = 0.08, float tc = 0.06, float ct = 0.02);
    virtual ~Dedot();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int filteringMode_;
    float lt_;
    float tl_;
    float tc_;
    float ct_;
};
