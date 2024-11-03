#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Rotate : public FilterBase {
public:
    /**
     * Rotate the input image.
     */
    /**
     * set angle (in radians)
     * Aliases: a
     * Type: String
     * Required: No
     * Default: 0
     */
    void setAngle(const std::string& value);
    std::string getAngle() const;

    /**
     * set output width expression
     * Aliases: ow
     * Type: String
     * Required: No
     * Default: iw
     */
    void setOut_w(const std::string& value);
    std::string getOut_w() const;

    /**
     * set output height expression
     * Aliases: oh
     * Type: String
     * Required: No
     * Default: ih
     */
    void setOut_h(const std::string& value);
    std::string getOut_h() const;

    /**
     * set background fill color
     * Aliases: c
     * Type: String
     * Required: No
     * Default: black
     */
    void setFillcolor(const std::string& value);
    std::string getFillcolor() const;

    /**
     * use bilinear interpolation
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setBilinear(bool value);
    bool getBilinear() const;

    Rotate(const std::string& angle = "0", const std::string& out_w = "iw", const std::string& out_h = "ih", const std::string& fillcolor = "black", bool bilinear = true);
    virtual ~Rotate();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string angle_;
    std::string out_w_;
    std::string out_h_;
    std::string fillcolor_;
    bool bilinear_;
};
