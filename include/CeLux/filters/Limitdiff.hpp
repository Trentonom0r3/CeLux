#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Limitdiff : public FilterBase {
public:
    /**
     * Apply filtering with limiting difference.
     */
    /**
     * set the threshold
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setThreshold(float value);
    float getThreshold() const;

    /**
     * set the elasticity
     * Type: Float
     * Required: No
     * Default: 2.00
     */
    void setElasticity(float value);
    float getElasticity() const;

    /**
     * enable reference stream
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setReference(bool value);
    bool getReference() const;

    /**
     * set the planes to filter
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Limitdiff(float threshold = 0.00, float elasticity = 2.00, bool reference = false, int planes = 15);
    virtual ~Limitdiff();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float threshold_;
    float elasticity_;
    bool reference_;
    int planes_;
};
