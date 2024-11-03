#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Xmedian : public FilterBase {
public:
    /**
     * Pick median pixels from several video inputs.
     */
    /**
     * set number of inputs
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setInputs(int value);
    int getInputs() const;

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

    Xmedian(int inputs = 3, int planes = 15, float percentile = 0.50);
    virtual ~Xmedian();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int inputs_;
    int planes_;
    float percentile_;
};
