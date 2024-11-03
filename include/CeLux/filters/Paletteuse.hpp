#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Paletteuse : public FilterBase {
public:
    /**
     * Use a palette to downsample an input video stream.
     */
    /**
     * select dithering mode
     * Unit: dithering_mode
     * Possible Values: bayer (1), heckbert (2), floyd_steinberg (3), sierra2 (4), sierra2_4a (5), sierra3 (6), burkes (7), atkinson (8)
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setDither(int value);
    int getDither() const;

    /**
     * set scale for bayer dithering
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setBayer_scale(int value);
    int getBayer_scale() const;

    /**
     * set frame difference mode
     * Unit: diff_mode
     * Possible Values: rectangle (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDiff_mode(int value);
    int getDiff_mode() const;

    /**
     * take new palette for each output frame
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setNew_(bool value);
    bool getNew_() const;

    /**
     * set the alpha threshold for transparency
     * Type: Integer
     * Required: No
     * Default: 128
     */
    void setAlpha_threshold(int value);
    int getAlpha_threshold() const;

    /**
     * save Graphviz graph of the kdtree in specified file
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setDebug_kdtree(const std::string& value);
    std::string getDebug_kdtree() const;

    Paletteuse(int dither = 5, int bayer_scale = 2, int diff_mode = 0, bool new_ = false, int alpha_threshold = 128, const std::string& debug_kdtree = "");
    virtual ~Paletteuse();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int dither_;
    int bayer_scale_;
    int diff_mode_;
    bool new__;
    int alpha_threshold_;
    std::string debug_kdtree_;
};
