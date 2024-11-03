#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Pixelize : public FilterBase {
public:
    /**
     * Pixelize video.
     */
    /**
     * set block width
     * Aliases: w
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setWidth(int value);
    int getWidth() const;

    /**
     * set block height
     * Aliases: h
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setHeight(int value);
    int getHeight() const;

    /**
     * set the pixelize mode
     * Aliases: m
     * Unit: mode
     * Possible Values: avg (0), min (1), max (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set what planes to filter
     * Aliases: p
     * Type: Flags
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Pixelize(int width = 16, int height = 16, int mode = 0, int planes = 15);
    virtual ~Pixelize();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int width_;
    int height_;
    int mode_;
    int planes_;
};
