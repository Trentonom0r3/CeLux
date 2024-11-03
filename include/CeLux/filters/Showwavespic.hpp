#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Showwavespic : public FilterBase {
public:
    /**
     * Convert input audio to a video output single picture.
     */
    /**
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 600x240
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * draw channels separately
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setSplit_channels(bool value);
    bool getSplit_channels() const;

    /**
     * set channels colors
     * Type: String
     * Required: No
     * Default: red|green|blue|yellow|orange|lime|pink|magenta|brown
     */
    void setColors(const std::string& value);
    std::string getColors() const;

    /**
     * set amplitude scale
     * Unit: scale
     * Possible Values: lin (0), log (1), sqrt (2), cbrt (3)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setScale(int value);
    int getScale() const;

    /**
     * set draw mode
     * Unit: draw
     * Possible Values: scale (0), full (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDraw(int value);
    int getDraw() const;

    /**
     * set filter mode
     * Unit: filter
     * Possible Values: average (0), peak (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFilter(int value);
    int getFilter() const;

    Showwavespic(std::pair<int, int> size = std::make_pair<int, int>(0, 1), bool split_channels = false, const std::string& colors = "red|green|blue|yellow|orange|lime|pink|magenta|brown", int scale = 0, int draw = 0, int filter = 0);
    virtual ~Showwavespic();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    bool split_channels_;
    std::string colors_;
    int scale_;
    int draw_;
    int filter_;
};
