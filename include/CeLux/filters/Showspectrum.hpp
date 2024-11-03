#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Showspectrum : public FilterBase {
public:
    /**
     * Convert input audio to a spectrum video output.
     */
    /**
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 640x512
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set sliding mode
     * Unit: slide
     * Possible Values: replace (0), scroll (1), fullframe (2), rscroll (3), lreplace (4)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setSlide(int value);
    int getSlide() const;

    /**
     * set channel display mode
     * Unit: mode
     * Possible Values: combined (0), separate (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set channel coloring
     * Unit: color
     * Possible Values: channel (0), intensity (1), rainbow (2), moreland (3), nebulae (4), fire (5), fiery (6), fruit (7), cool (8), magma (9), green (10), viridis (11), plasma (12), cividis (13), terrain (14)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setColor(int value);
    int getColor() const;

    /**
     * set display scale
     * Unit: scale
     * Possible Values: lin (0), sqrt (1), cbrt (2), log (3), 4thrt (4), 5thrt (5)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setScale(int value);
    int getScale() const;

    /**
     * set frequency scale
     * Unit: fscale
     * Possible Values: lin (0), log (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFscale(int value);
    int getFscale() const;

    /**
     * color saturation multiplier
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setSaturation(float value);
    float getSaturation() const;

    /**
     * set window function
     * Unit: win_func
     * Possible Values: rect (0), bartlett (4), hann (1), hanning (1), hamming (2), blackman (3), welch (5), flattop (6), bharris (7), bnuttall (8), bhann (11), sine (9), nuttall (10), lanczos (12), gauss (13), tukey (14), dolph (15), cauchy (16), parzen (17), poisson (18), bohman (19), kaiser (20)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setWin_func(int value);
    int getWin_func() const;

    /**
     * set orientation
     * Unit: orientation
     * Possible Values: vertical (0), horizontal (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOrientation(int value);
    int getOrientation() const;

    /**
     * set window overlap
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setOverlap(float value);
    float getOverlap() const;

    /**
     * set scale gain
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setGain(float value);
    float getGain() const;

    /**
     * set data mode
     * Unit: data
     * Possible Values: magnitude (0), phase (1), uphase (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setData(int value);
    int getData() const;

    /**
     * color rotation
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRotation(float value);
    float getRotation() const;

    /**
     * start frequency
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setStart(int value);
    int getStart() const;

    /**
     * stop frequency
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setStop(int value);
    int getStop() const;

    /**
     * set video rate
     * Type: String
     * Required: No
     * Default: auto
     */
    void setFps(const std::string& value);
    std::string getFps() const;

    /**
     * draw legend
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setLegend(bool value);
    bool getLegend() const;

    /**
     * set dynamic range in dBFS
     * Type: Float
     * Required: No
     * Default: 120.00
     */
    void setDrange(float value);
    float getDrange() const;

    /**
     * set upper limit in dBFS
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setLimit(float value);
    float getLimit() const;

    /**
     * set opacity strength
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setOpacity(float value);
    float getOpacity() const;

    Showspectrum(std::pair<int, int> size = std::make_pair<int, int>(0, 1), int slide = 0, int mode = 0, int color = 0, int scale = 1, int fscale = 0, float saturation = 1.00, int win_func = 1, int orientation = 0, float overlap = 0.00, float gain = 1.00, int data = 0, float rotation = 0.00, int start = 0, int stop = 0, const std::string& fps = "auto", bool legend = false, float drange = 120.00, float limit = 0.00, float opacity = 1.00);
    virtual ~Showspectrum();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    int slide_;
    int mode_;
    int color_;
    int scale_;
    int fscale_;
    float saturation_;
    int win_func_;
    int orientation_;
    float overlap_;
    float gain_;
    int data_;
    float rotation_;
    int start_;
    int stop_;
    std::string fps_;
    bool legend_;
    float drange_;
    float limit_;
    float opacity_;
};
