#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Maskedclamp : public FilterBase {
public:
    /**
     * Clamp first stream with second stream and third stream.
     */
    /**
     * set undershoot
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setUndershoot(int value);
    int getUndershoot() const;

    /**
     * set overshoot
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOvershoot(int value);
    int getOvershoot() const;

    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Maskedclamp(int undershoot = 0, int overshoot = 0, int planes = 15);
    virtual ~Maskedclamp();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int undershoot_;
    int overshoot_;
    int planes_;
};
