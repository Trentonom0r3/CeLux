#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Pseudocolor : public FilterBase {
public:
    /**
     * Make pseudocolored video frames.
     */
    /**
     * set component #0 expression
     * Type: String
     * Required: No
     * Default: val
     */
    void setC0(const std::string& value);
    std::string getC0() const;

    /**
     * set component #1 expression
     * Type: String
     * Required: No
     * Default: val
     */
    void setC1(const std::string& value);
    std::string getC1() const;

    /**
     * set component #2 expression
     * Type: String
     * Required: No
     * Default: val
     */
    void setC2(const std::string& value);
    std::string getC2() const;

    /**
     * set component #3 expression
     * Type: String
     * Required: No
     * Default: val
     */
    void setC3(const std::string& value);
    std::string getC3() const;

    /**
     * set component as base
     * Aliases: i
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setIndex(int value);
    int getIndex() const;

    /**
     * set preset
     * Aliases: p
     * Unit: preset
     * Possible Values: none (-1), magma (0), inferno (1), plasma (2), viridis (3), turbo (4), cividis (5), range1 (6), range2 (7), shadows (8), highlights (9), solar (10), nominal (11), preferred (12), total (13), spectral (14), cool (15), heat (16), fiery (17), blues (18), green (19), helix (20)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setPreset(int value);
    int getPreset() const;

    /**
     * set pseudocolor opacity
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setOpacity(float value);
    float getOpacity() const;

    Pseudocolor(const std::string& c0 = "val", const std::string& c1 = "val", const std::string& c2 = "val", const std::string& c3 = "val", int index = 0, int preset = -1, float opacity = 1.00);
    virtual ~Pseudocolor();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string c0_;
    std::string c1_;
    std::string c2_;
    std::string c3_;
    int index_;
    int preset_;
    float opacity_;
};
