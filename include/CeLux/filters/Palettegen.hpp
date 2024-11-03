#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Palettegen : public FilterBase {
public:
    /**
     * Find the optimal palette for a given stream.
     */
    /**
     * set the maximum number of colors to use in the palette
     * Type: Integer
     * Required: No
     * Default: 256
     */
    void setMax_colors(int value);
    int getMax_colors() const;

    /**
     * reserve a palette entry for transparency
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setReserve_transparent(bool value);
    bool getReserve_transparent() const;

    /**
     * set a background color for transparency
     * Type: Color
     * Required: No
     * Default: lime
     */
    void setTransparency_color(const std::string& value);
    std::string getTransparency_color() const;

    /**
     * set statistics mode
     * Unit: mode
     * Possible Values: full (0), diff (1), single (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setStats_mode(int value);
    int getStats_mode() const;

    Palettegen(int max_colors = 256, bool reserve_transparent = true, const std::string& transparency_color = "lime", int stats_mode = 0);
    virtual ~Palettegen();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int max_colors_;
    bool reserve_transparent_;
    std::string transparency_color_;
    int stats_mode_;
};
