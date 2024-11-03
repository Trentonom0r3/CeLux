#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Geq : public FilterBase {
public:
    /**
     * Apply generic equation to each pixel.
     */
    /**
     * set luminance expression
     * Aliases: lum
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setLum_expr(const std::string& value);
    std::string getLum_expr() const;

    /**
     * set chroma blue expression
     * Aliases: cb
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setCb_expr(const std::string& value);
    std::string getCb_expr() const;

    /**
     * set chroma red expression
     * Aliases: cr
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setCr_expr(const std::string& value);
    std::string getCr_expr() const;

    /**
     * set alpha expression
     * Aliases: a
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setAlpha_expr(const std::string& value);
    std::string getAlpha_expr() const;

    /**
     * set red expression
     * Aliases: r
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setRed_expr(const std::string& value);
    std::string getRed_expr() const;

    /**
     * set green expression
     * Aliases: g
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setGreen_expr(const std::string& value);
    std::string getGreen_expr() const;

    /**
     * set blue expression
     * Aliases: b
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setBlue_expr(const std::string& value);
    std::string getBlue_expr() const;

    /**
     * set interpolation method
     * Aliases: i
     * Unit: interp
     * Possible Values: nearest (0), n (0), bilinear (1), b (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setInterpolation(int value);
    int getInterpolation() const;

    Geq(const std::string& lum_expr = "", const std::string& cb_expr = "", const std::string& cr_expr = "", const std::string& alpha_expr = "", const std::string& red_expr = "", const std::string& green_expr = "", const std::string& blue_expr = "", int interpolation = 1);
    virtual ~Geq();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string lum_expr_;
    std::string cb_expr_;
    std::string cr_expr_;
    std::string alpha_expr_;
    std::string red_expr_;
    std::string green_expr_;
    std::string blue_expr_;
    int interpolation_;
};
