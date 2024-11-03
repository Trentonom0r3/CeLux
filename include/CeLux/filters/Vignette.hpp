#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Vignette : public FilterBase {
public:
    /**
     * Make or reverse a vignette effect.
     */
    /**
     * set lens angle
     * Aliases: a
     * Type: String
     * Required: No
     * Default: PI/5
     */
    void setAngle(const std::string& value);
    std::string getAngle() const;

    /**
     * set circle center position on x-axis
     * Type: String
     * Required: No
     * Default: w/2
     */
    void setX0(const std::string& value);
    std::string getX0() const;

    /**
     * set circle center position on y-axis
     * Type: String
     * Required: No
     * Default: h/2
     */
    void setY0(const std::string& value);
    std::string getY0() const;

    /**
     * set forward/backward mode
     * Unit: mode
     * Possible Values: forward (0), backward (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * specify when to evaluate expressions
     * Unit: eval
     * Possible Values: init (0), frame (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setEval(int value);
    int getEval() const;

    /**
     * set dithering
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setDither(bool value);
    bool getDither() const;

    /**
     * set aspect ratio
     * Type: Rational
     * Required: No
     * Default: 0
     */
    void setAspect(const std::pair<int, int>& value);
    std::pair<int, int> getAspect() const;

    Vignette(const std::string& angle = "PI/5", const std::string& x0 = "w/2", const std::string& y0 = "h/2", int mode = 0, int eval = 0, bool dither = true, std::pair<int, int> aspect = std::make_pair<int, int>(0, 1));
    virtual ~Vignette();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string angle_;
    std::string x0_;
    std::string y0_;
    int mode_;
    int eval_;
    bool dither_;
    std::pair<int, int> aspect_;
};
