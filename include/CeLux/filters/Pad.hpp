#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Pad : public FilterBase {
public:
    /**
     * Pad the input video.
     */
    /**
     * set the pad area width expression
     * Aliases: w
     * Type: String
     * Required: No
     * Default: iw
     */
    void setWidth(const std::string& value);
    std::string getWidth() const;

    /**
     * set the pad area height expression
     * Aliases: h
     * Type: String
     * Required: No
     * Default: ih
     */
    void setHeight(const std::string& value);
    std::string getHeight() const;

    /**
     * set the x offset expression for the input image position
     * Type: String
     * Required: No
     * Default: 0
     */
    void setXOffsetForTheInputImagePosition(const std::string& value);
    std::string getXOffsetForTheInputImagePosition() const;

    /**
     * set the y offset expression for the input image position
     * Type: String
     * Required: No
     * Default: 0
     */
    void setYOffsetForTheInputImagePosition(const std::string& value);
    std::string getYOffsetForTheInputImagePosition() const;

    /**
     * set the color of the padded area border
     * Type: Color
     * Required: No
     * Default: black
     */
    void setColor(const std::string& value);
    std::string getColor() const;

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
     * pad to fit an aspect instead of a resolution
     * Type: Rational
     * Required: Yes
     * Default: No Default
     */
    void setAspect(const std::pair<int, int>& value);
    std::pair<int, int> getAspect() const;

    Pad(const std::string& width = "iw", const std::string& height = "ih", const std::string& xOffsetForTheInputImagePosition = "0", const std::string& yOffsetForTheInputImagePosition = "0", const std::string& color = "black", int eval = 0, std::pair<int, int> aspect = std::make_pair<int, int>(0, 1));
    virtual ~Pad();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string width_;
    std::string height_;
    std::string xOffsetForTheInputImagePosition_;
    std::string yOffsetForTheInputImagePosition_;
    std::string color_;
    int eval_;
    std::pair<int, int> aspect_;
};
