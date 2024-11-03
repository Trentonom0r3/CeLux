#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Datascope : public FilterBase {
public:
    /**
     * Video data analysis.
     */
    /**
     * set output size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: hd720
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set x offset
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setXOffset(int value);
    int getXOffset() const;

    /**
     * set y offset
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setYOffset(int value);
    int getYOffset() const;

    /**
     * set scope mode
     * Unit: mode
     * Possible Values: mono (0), color (1), color2 (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * draw column/row numbers
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setAxis(bool value);
    bool getAxis() const;

    /**
     * set background opacity
     * Type: Float
     * Required: No
     * Default: 0.75
     */
    void setOpacity(float value);
    float getOpacity() const;

    /**
     * set display number format
     * Unit: format
     * Possible Values: hex (0), dec (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFormat(int value);
    int getFormat() const;

    /**
     * set components to display
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setComponents(int value);
    int getComponents() const;

    Datascope(std::pair<int, int> size = std::make_pair<int, int>(0, 1), int xOffset = 0, int yOffset = 0, int mode = 0, bool axis = false, float opacity = 0.75, int format = 0, int components = 15);
    virtual ~Datascope();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    int xOffset_;
    int yOffset_;
    int mode_;
    bool axis_;
    float opacity_;
    int format_;
    int components_;
};
