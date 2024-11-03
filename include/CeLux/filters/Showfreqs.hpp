#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Showfreqs : public FilterBase {
public:
    /**
     * Convert input audio to a frequencies video output.
     */
    /**
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 1024x512
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set video rate
     * Aliases: r
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    /**
     * set display mode
     * Unit: mode
     * Possible Values: line (0), bar (1), dot (2)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set amplitude scale
     * Unit: ascale
     * Possible Values: lin (0), sqrt (1), cbrt (2), log (3)
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setAscale(int value);
    int getAscale() const;

    /**
     * set frequency scale
     * Unit: fscale
     * Possible Values: lin (0), log (1), rlog (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFscale(int value);
    int getFscale() const;

    /**
     * set window size
     * Type: Integer
     * Required: No
     * Default: 2048
     */
    void setWin_size(int value);
    int getWin_size() const;

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
     * set window overlap
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setOverlap(float value);
    float getOverlap() const;

    /**
     * set time averaging
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setAveraging(int value);
    int getAveraging() const;

    /**
     * set channels colors
     * Type: String
     * Required: No
     * Default: red|green|blue|yellow|orange|lime|pink|magenta|brown
     */
    void setColors(const std::string& value);
    std::string getColors() const;

    /**
     * set channel mode
     * Unit: cmode
     * Possible Values: combined (0), separate (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCmode(int value);
    int getCmode() const;

    /**
     * set minimum amplitude
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setMinamp(float value);
    float getMinamp() const;

    /**
     * set data mode
     * Unit: data
     * Possible Values: magnitude (0), phase (1), delay (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setData(int value);
    int getData() const;

    /**
     * set channels to draw
     * Type: String
     * Required: No
     * Default: all
     */
    void setChannels(const std::string& value);
    std::string getChannels() const;

    Showfreqs(std::pair<int, int> size = std::make_pair<int, int>(0, 1), std::pair<int, int> rate = std::make_pair<int, int>(0, 1), int mode = 1, int ascale = 3, int fscale = 0, int win_size = 2048, int win_func = 1, float overlap = 1.00, int averaging = 1, const std::string& colors = "red|green|blue|yellow|orange|lime|pink|magenta|brown", int cmode = 0, float minamp = 0.00, int data = 0, const std::string& channels = "all");
    virtual ~Showfreqs();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    std::pair<int, int> rate_;
    int mode_;
    int ascale_;
    int fscale_;
    int win_size_;
    int win_func_;
    float overlap_;
    int averaging_;
    std::string colors_;
    int cmode_;
    float minamp_;
    int data_;
    std::string channels_;
};
