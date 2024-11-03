#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Ahistogram : public FilterBase {
public:
    /**
     * Convert input audio to histogram video output.
     */
    /**
     * set method to display channels
     * Unit: dmode
     * Possible Values: single (0), separate (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDmode(int value);
    int getDmode() const;

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
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: hd720
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set display scale
     * Unit: scale
     * Possible Values: log (3), sqrt (1), cbrt (2), lin (0), rlog (4)
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setScale(int value);
    int getScale() const;

    /**
     * set amplitude scale
     * Unit: ascale
     * Possible Values: log (1), lin (0)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setAscale(int value);
    int getAscale() const;

    /**
     * how much frames to accumulate
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setAcount(int value);
    int getAcount() const;

    /**
     * set histogram ratio of window height
     * Type: Float
     * Required: No
     * Default: 0.10
     */
    void setRheight(float value);
    float getRheight() const;

    /**
     * set sonogram sliding
     * Unit: slide
     * Possible Values: replace (0), scroll (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setSlide(int value);
    int getSlide() const;

    /**
     * set histograms mode
     * Unit: hmode
     * Possible Values: abs (0), sign (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setHmode(int value);
    int getHmode() const;

    Ahistogram(int dmode = 0, std::pair<int, int> rate = std::make_pair<int, int>(0, 1), std::pair<int, int> size = std::make_pair<int, int>(0, 1), int scale = 3, int ascale = 1, int acount = 1, float rheight = 0.10, int slide = 0, int hmode = 0);
    virtual ~Ahistogram();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int dmode_;
    std::pair<int, int> rate_;
    std::pair<int, int> size_;
    int scale_;
    int ascale_;
    int acount_;
    float rheight_;
    int slide_;
    int hmode_;
};
