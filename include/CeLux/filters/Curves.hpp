#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Curves : public FilterBase {
public:
    /**
     * Adjust components curves.
     */
    /**
     * select a color curves preset
     * Unit: preset_name
     * Possible Values: none (0), color_negative (1), cross_process (2), darker (3), increase_contrast (4), lighter (5), linear_contrast (6), medium_contrast (7), negative (8), strong_contrast (9), vintage (10)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPreset(int value);
    int getPreset() const;

    /**
     * set master points coordinates
     * Aliases: m
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setMaster(const std::string& value);
    std::string getMaster() const;

    /**
     * set red points coordinates
     * Aliases: r
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setRed(const std::string& value);
    std::string getRed() const;

    /**
     * set green points coordinates
     * Aliases: g
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setGreen(const std::string& value);
    std::string getGreen() const;

    /**
     * set blue points coordinates
     * Aliases: b
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setBlue(const std::string& value);
    std::string getBlue() const;

    /**
     * set points coordinates for all components
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setAll(const std::string& value);
    std::string getAll() const;

    /**
     * set Photoshop curves file name
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setPsfile(const std::string& value);
    std::string getPsfile() const;

    /**
     * save Gnuplot script of the curves in specified file
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setPlot(const std::string& value);
    std::string getPlot() const;

    /**
     * specify the kind of interpolation
     * Unit: interp_name
     * Possible Values: natural (0), pchip (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setInterp(int value);
    int getInterp() const;

    Curves(int preset = 0, const std::string& master = "", const std::string& red = "", const std::string& green = "", const std::string& blue = "", const std::string& all = "", const std::string& psfile = "", const std::string& plot = "", int interp = 0);
    virtual ~Curves();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int preset_;
    std::string master_;
    std::string red_;
    std::string green_;
    std::string blue_;
    std::string all_;
    std::string psfile_;
    std::string plot_;
    int interp_;
};
