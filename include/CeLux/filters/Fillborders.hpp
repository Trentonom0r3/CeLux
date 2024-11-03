#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Fillborders : public FilterBase {
public:
    /**
     * Fill borders of the input video.
     */
    /**
     * set the left fill border
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setLeft(int value);
    int getLeft() const;

    /**
     * set the right fill border
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setRight(int value);
    int getRight() const;

    /**
     * set the top fill border
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setTop(int value);
    int getTop() const;

    /**
     * set the bottom fill border
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setBottom(int value);
    int getBottom() const;

    /**
     * set the fill borders mode
     * Unit: mode
     * Possible Values: smear (0), mirror (1), fixed (2), reflect (3), wrap (4), fade (5), margins (6)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set the color for the fixed/fade mode
     * Type: Color
     * Required: No
     * Default: black
     */
    void setColor(const std::string& value);
    std::string getColor() const;

    Fillborders(int left = 0, int right = 0, int top = 0, int bottom = 0, int mode = 0, const std::string& color = "black");
    virtual ~Fillborders();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int left_;
    int right_;
    int top_;
    int bottom_;
    int mode_;
    std::string color_;
};
