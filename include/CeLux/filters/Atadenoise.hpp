#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Atadenoise : public FilterBase {
public:
    /**
     * Apply an Adaptive Temporal Averaging Denoiser.
     */
    /**
     * set threshold A for 1st plane
     * Type: Float
     * Required: No
     * Default: 0.02
     */
    void set_0a(float value);
    float get_0a() const;

    /**
     * set threshold B for 1st plane
     * Type: Float
     * Required: No
     * Default: 0.04
     */
    void set_0b(float value);
    float get_0b() const;

    /**
     * set threshold A for 2nd plane
     * Type: Float
     * Required: No
     * Default: 0.02
     */
    void set_1a(float value);
    float get_1a() const;

    /**
     * set threshold B for 2nd plane
     * Type: Float
     * Required: No
     * Default: 0.04
     */
    void set_1b(float value);
    float get_1b() const;

    /**
     * set threshold A for 3rd plane
     * Type: Float
     * Required: No
     * Default: 0.02
     */
    void set_2a(float value);
    float get_2a() const;

    /**
     * set threshold B for 3rd plane
     * Type: Float
     * Required: No
     * Default: 0.04
     */
    void set_2b(float value);
    float get_2b() const;

    /**
     * set how many frames to use
     * Type: Integer
     * Required: No
     * Default: 9
     */
    void setHowManyFramesToUse(int value);
    int getHowManyFramesToUse() const;

    /**
     * set what planes to filter
     * Type: Flags
     * Required: No
     * Default: 7
     */
    void setWhatPlanesToFilter(int value);
    int getWhatPlanesToFilter() const;

    /**
     * set variant of algorithm
     * Unit: a
     * Possible Values: p (0), s (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setVariantOfAlgorithm(int value);
    int getVariantOfAlgorithm() const;

    /**
     * set sigma for 1st plane
     * Type: Float
     * Required: No
     * Default: 32767.00
     */
    void set_0s(float value);
    float get_0s() const;

    /**
     * set sigma for 2nd plane
     * Type: Float
     * Required: No
     * Default: 32767.00
     */
    void set_1s(float value);
    float get_1s() const;

    /**
     * set sigma for 3rd plane
     * Type: Float
     * Required: No
     * Default: 32767.00
     */
    void set_2s(float value);
    float get_2s() const;

    Atadenoise(float _0a = 0.02, float _0b = 0.04, float _1a = 0.02, float _1b = 0.04, float _2a = 0.02, float _2b = 0.04, int howManyFramesToUse = 9, int whatPlanesToFilter = 7, int variantOfAlgorithm = 0, float _0s = 32767.00, float _1s = 32767.00, float _2s = 32767.00);
    virtual ~Atadenoise();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float _0a_;
    float _0b_;
    float _1a_;
    float _1b_;
    float _2a_;
    float _2b_;
    int howManyFramesToUse_;
    int whatPlanesToFilter_;
    int variantOfAlgorithm_;
    float _0s_;
    float _1s_;
    float _2s_;
};
