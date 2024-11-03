#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Showcwt : public FilterBase {
public:
    /**
     * Convert input audio to a CWT (Continuous Wavelet Transform) spectrum video output.
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
     * set video rate
     * Aliases: r
     * Type: String
     * Required: No
     * Default: 25
     */
    void setRate(const std::string& value);
    std::string getRate() const;

    /**
     * set frequency scale
     * Unit: scale
     * Possible Values: linear (0), log (1), bark (2), mel (3), erbs (4), sqrt (5), cbrt (6), qdrt (7), fm (8)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setScale(int value);
    int getScale() const;

    /**
     * set intensity scale
     * Unit: iscale
     * Possible Values: linear (1), log (0), sqrt (2), cbrt (3), qdrt (4)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setIscale(int value);
    int getIscale() const;

    /**
     * set minimum frequency
     * Type: Float
     * Required: No
     * Default: 20.00
     */
    void setMin(float value);
    float getMin() const;

    /**
     * set maximum frequency
     * Type: Float
     * Required: No
     * Default: 20000.00
     */
    void setMax(float value);
    float getMax() const;

    /**
     * set minimum intensity
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setImin(float value);
    float getImin() const;

    /**
     * set maximum intensity
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setImax(float value);
    float getImax() const;

    /**
     * set logarithmic basis
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setLogb(float value);
    float getLogb() const;

    /**
     * set frequency deviation
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setDeviation(float value);
    float getDeviation() const;

    /**
     * set pixels per second
     * Type: Integer
     * Required: No
     * Default: 64
     */
    void setPps(int value);
    int getPps() const;

    /**
     * set output mode
     * Unit: mode
     * Possible Values: magnitude (0), phase (1), magphase (2), channel (3), stereo (4)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set slide mode
     * Unit: slide
     * Possible Values: replace (0), scroll (1), frame (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setSlide(int value);
    int getSlide() const;

    /**
     * set direction mode
     * Unit: direction
     * Possible Values: lr (0), rl (1), ud (2), du (3)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDirection(int value);
    int getDirection() const;

    /**
     * set bargraph ratio
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBar(float value);
    float getBar() const;

    /**
     * set color rotation
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRotation(float value);
    float getRotation() const;

    Showcwt(std::pair<int, int> size = std::make_pair<int, int>(0, 1), const std::string& rate = "25", int scale = 0, int iscale = 0, float min = 20.00, float max = 20000.00, float imin = 0.00, float imax = 1.00, float logb = 0.00, float deviation = 1.00, int pps = 64, int mode = 0, int slide = 0, int direction = 0, float bar = 0.00, float rotation = 0.00);
    virtual ~Showcwt();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    std::string rate_;
    int scale_;
    int iscale_;
    float min_;
    float max_;
    float imin_;
    float imax_;
    float logb_;
    float deviation_;
    int pps_;
    int mode_;
    int slide_;
    int direction_;
    float bar_;
    float rotation_;
};
