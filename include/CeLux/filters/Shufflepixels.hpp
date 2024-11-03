#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Shufflepixels : public FilterBase {
public:
    /**
     * Shuffle video pixels.
     */
    /**
     * set shuffle direction
     * Aliases: d
     * Unit: dir
     * Possible Values: forward (0), inverse (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDirection(int value);
    int getDirection() const;

    /**
     * set shuffle mode
     * Aliases: m
     * Unit: mode
     * Possible Values: horizontal (0), vertical (1), block (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set block width
     * Aliases: w
     * Type: Integer
     * Required: No
     * Default: 10
     */
    void setWidth(int value);
    int getWidth() const;

    /**
     * set block height
     * Aliases: h
     * Type: Integer
     * Required: No
     * Default: 10
     */
    void setHeight(int value);
    int getHeight() const;

    /**
     * set random seed
     * Aliases: s
     * Type: Integer64
     * Required: No
     * Default: -1
     */
    void setSeed(int64_t value);
    int64_t getSeed() const;

    Shufflepixels(int direction = 0, int mode = 0, int width = 10, int height = 10, int64_t seed = 0);
    virtual ~Shufflepixels();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int direction_;
    int mode_;
    int width_;
    int height_;
    int64_t seed_;
};
