#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Erosion : public FilterBase {
public:
    /**
     * Apply erosion effect.
     */
    /**
     * set coordinates
     * Type: Integer
     * Required: No
     * Default: 255
     */
    void setCoordinates(int value);
    int getCoordinates() const;

    /**
     * set threshold for 1st plane
     * Type: Integer
     * Required: No
     * Default: 65535
     */
    void setThreshold0(int value);
    int getThreshold0() const;

    /**
     * set threshold for 2nd plane
     * Type: Integer
     * Required: No
     * Default: 65535
     */
    void setThreshold1(int value);
    int getThreshold1() const;

    /**
     * set threshold for 3rd plane
     * Type: Integer
     * Required: No
     * Default: 65535
     */
    void setThreshold2(int value);
    int getThreshold2() const;

    /**
     * set threshold for 4th plane
     * Type: Integer
     * Required: No
     * Default: 65535
     */
    void setThreshold3(int value);
    int getThreshold3() const;

    Erosion(int coordinates = 255, int threshold0 = 65535, int threshold1 = 65535, int threshold2 = 65535, int threshold3 = 65535);
    virtual ~Erosion();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int coordinates_;
    int threshold0_;
    int threshold1_;
    int threshold2_;
    int threshold3_;
};
