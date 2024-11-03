#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Floodfill : public FilterBase {
public:
    /**
     * Fill area with same color with another color.
     */
    /**
     * set pixel x coordinate
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPixelXCoordinate(int value);
    int getPixelXCoordinate() const;

    /**
     * set pixel y coordinate
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPixelYCoordinate(int value);
    int getPixelYCoordinate() const;

    /**
     * set source #0 component value
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setS0(int value);
    int getS0() const;

    /**
     * set source #1 component value
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setS1(int value);
    int getS1() const;

    /**
     * set source #2 component value
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setS2(int value);
    int getS2() const;

    /**
     * set source #3 component value
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setS3(int value);
    int getS3() const;

    /**
     * set destination #0 component value
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setD0(int value);
    int getD0() const;

    /**
     * set destination #1 component value
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setD1(int value);
    int getD1() const;

    /**
     * set destination #2 component value
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setD2(int value);
    int getD2() const;

    /**
     * set destination #3 component value
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setD3(int value);
    int getD3() const;

    Floodfill(int pixelXCoordinate = 0, int pixelYCoordinate = 0, int s0 = 0, int s1 = 0, int s2 = 0, int s3 = 0, int d0 = 0, int d1 = 0, int d2 = 0, int d3 = 0);
    virtual ~Floodfill();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int pixelXCoordinate_;
    int pixelYCoordinate_;
    int s0_;
    int s1_;
    int s2_;
    int s3_;
    int d0_;
    int d1_;
    int d2_;
    int d3_;
};
