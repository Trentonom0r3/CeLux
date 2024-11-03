#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Backgroundkey : public FilterBase {
public:
    /**
     * Turns a static background into transparency.
     */
    /**
     * set the scene change threshold
     * Type: Float
     * Required: No
     * Default: 0.08
     */
    void setThreshold(float value);
    float getThreshold() const;

    /**
     * set the similarity
     * Type: Float
     * Required: No
     * Default: 0.10
     */
    void setSimilarity(float value);
    float getSimilarity() const;

    /**
     * set the blend value
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBlend(float value);
    float getBlend() const;

    Backgroundkey(float threshold = 0.08, float similarity = 0.10, float blend = 0.00);
    virtual ~Backgroundkey();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float threshold_;
    float similarity_;
    float blend_;
};
