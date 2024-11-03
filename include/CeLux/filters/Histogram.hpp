#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Histogram : public FilterBase {
public:
    /**
     * Compute and draw a histogram.
     */
    /**
     * set level height
     * Type: Integer
     * Required: No
     * Default: 200
     */
    void setLevel_height(int value);
    int getLevel_height() const;

    /**
     * set scale height
     * Type: Integer
     * Required: No
     * Default: 12
     */
    void setScale_height(int value);
    int getScale_height() const;

    /**
     * set display mode
     * Aliases: d
     * Unit: display_mode
     * Possible Values: overlay (0), parade (1), stack (2)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setDisplay_mode(int value);
    int getDisplay_mode() const;

    /**
     * set levels mode
     * Aliases: m
     * Unit: levels_mode
     * Possible Values: linear (0), logarithmic (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setLevels_mode(int value);
    int getLevels_mode() const;

    /**
     * set color components to display
     * Aliases: c
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setComponents(int value);
    int getComponents() const;

    /**
     * set foreground opacity
     * Aliases: f
     * Type: Float
     * Required: No
     * Default: 0.70
     */
    void setFgopacity(float value);
    float getFgopacity() const;

    /**
     * set background opacity
     * Aliases: b
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setBgopacity(float value);
    float getBgopacity() const;

    /**
     * set colors mode
     * Aliases: l
     * Unit: colors_mode
     * Possible Values: whiteonblack (0), blackonwhite (1), whiteongray (2), blackongray (3), coloronblack (4), coloronwhite (5), colorongray (6), blackoncolor (7), whiteoncolor (8), grayoncolor (9)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setColors_mode(int value);
    int getColors_mode() const;

    Histogram(int level_height = 200, int scale_height = 12, int display_mode = 2, int levels_mode = 0, int components = 7, float fgopacity = 0.70, float bgopacity = 0.50, int colors_mode = 0);
    virtual ~Histogram();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int level_height_;
    int scale_height_;
    int display_mode_;
    int levels_mode_;
    int components_;
    float fgopacity_;
    float bgopacity_;
    int colors_mode_;
};
