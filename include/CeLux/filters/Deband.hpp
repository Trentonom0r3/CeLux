#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Deband : public FilterBase {
public:
    /**
     * Debands video.
     */
    /**
     * set 1st plane threshold
     * Type: Float
     * Required: No
     * Default: 0.02
     */
    void set_1thr(float value);
    float get_1thr() const;

    /**
     * set 2nd plane threshold
     * Type: Float
     * Required: No
     * Default: 0.02
     */
    void set_2thr(float value);
    float get_2thr() const;

    /**
     * set 3rd plane threshold
     * Type: Float
     * Required: No
     * Default: 0.02
     */
    void set_3thr(float value);
    float get_3thr() const;

    /**
     * set 4th plane threshold
     * Type: Float
     * Required: No
     * Default: 0.02
     */
    void set_4thr(float value);
    float get_4thr() const;

    /**
     * set range
     * Aliases: r
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setRange(int value);
    int getRange() const;

    /**
     * set direction
     * Aliases: d
     * Type: Float
     * Required: No
     * Default: 6.28
     */
    void setDirection(float value);
    float getDirection() const;

    /**
     * set blur
     * Aliases: b
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setBlur(bool value);
    bool getBlur() const;

    /**
     * set plane coupling
     * Aliases: c
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setCoupling(bool value);
    bool getCoupling() const;

    Deband(float _1thr = 0.02, float _2thr = 0.02, float _3thr = 0.02, float _4thr = 0.02, int range = 16, float direction = 6.28, bool blur = true, bool coupling = false);
    virtual ~Deband();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float _1thr_;
    float _2thr_;
    float _3thr_;
    float _4thr_;
    int range_;
    float direction_;
    bool blur_;
    bool coupling_;
};
