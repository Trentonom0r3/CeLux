#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Thistogram : public FilterBase {
public:
    /**
     * Compute and draw a temporal histogram.
     */
    /**
     * set width
     * Aliases: w
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setWidth(int value);
    int getWidth() const;

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
     * set background opacity
     * Aliases: b
     * Type: Float
     * Required: No
     * Default: 0.90
     */
    void setBgopacity(float value);
    float getBgopacity() const;

    /**
     * display envelope
     * Aliases: e
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setEnvelope(bool value);
    bool getEnvelope() const;

    /**
     * set envelope color
     * Aliases: ec
     * Type: Color
     * Required: No
     * Default: gold
     */
    void setEcolor(const std::string& value);
    std::string getEcolor() const;

    /**
     * set slide mode
     * Unit: slide
     * Possible Values: frame (0), replace (1), scroll (2), rscroll (3), picture (4)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setSlide(int value);
    int getSlide() const;

    Thistogram(int width = 0, int display_mode = 2, int levels_mode = 0, int components = 7, float bgopacity = 0.90, bool envelope = false, const std::string& ecolor = "gold", int slide = 1);
    virtual ~Thistogram();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int width_;
    int display_mode_;
    int levels_mode_;
    int components_;
    float bgopacity_;
    bool envelope_;
    std::string ecolor_;
    int slide_;
};
