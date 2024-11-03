#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Showspatial : public FilterBase {
public:
    /**
     * Convert input audio to a spatial video output.
     */
    /**
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 512x512
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set window size
     * Type: Integer
     * Required: No
     * Default: 4096
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
     * set video rate
     * Aliases: r
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    Showspatial(std::pair<int, int> size = std::make_pair<int, int>(0, 1), int win_size = 4096, int win_func = 1, std::pair<int, int> rate = std::make_pair<int, int>(0, 1));
    virtual ~Showspatial();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    int win_size_;
    int win_func_;
    std::pair<int, int> rate_;
};
