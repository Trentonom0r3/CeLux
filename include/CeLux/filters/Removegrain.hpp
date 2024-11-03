#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Removegrain : public FilterBase {
public:
    /**
     * Remove grain.
     */
    /**
     * set mode for 1st plane
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setM0(int value);
    int getM0() const;

    /**
     * set mode for 2nd plane
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setM1(int value);
    int getM1() const;

    /**
     * set mode for 3rd plane
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setM2(int value);
    int getM2() const;

    /**
     * set mode for 4th plane
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setM3(int value);
    int getM3() const;

    Removegrain(int m0 = 0, int m1 = 0, int m2 = 0, int m3 = 0);
    virtual ~Removegrain();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int m0_;
    int m1_;
    int m2_;
    int m3_;
};
