#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Drawgrid : public FilterBase {
public:
    /**
     * Draw a colored grid on the input video.
     */
    /**
     * set horizontal offset
     * Type: String
     * Required: No
     * Default: 0
     */
    void setHorizontalOffset(const std::string& value);
    std::string getHorizontalOffset() const;

    /**
     * set vertical offset
     * Type: String
     * Required: No
     * Default: 0
     */
    void setVerticalOffset(const std::string& value);
    std::string getVerticalOffset() const;

    /**
     * set width of grid cell
     * Aliases: w
     * Type: String
     * Required: No
     * Default: 0
     */
    void setWidth(const std::string& value);
    std::string getWidth() const;

    /**
     * set height of grid cell
     * Aliases: h
     * Type: String
     * Required: No
     * Default: 0
     */
    void setHeight(const std::string& value);
    std::string getHeight() const;

    /**
     * set color of the grid
     * Aliases: c
     * Type: String
     * Required: No
     * Default: black
     */
    void setColor(const std::string& value);
    std::string getColor() const;

    /**
     * set grid line thickness
     * Aliases: t
     * Type: String
     * Required: No
     * Default: 1
     */
    void setThickness(const std::string& value);
    std::string getThickness() const;

    /**
     * replace color & alpha
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setReplace(bool value);
    bool getReplace() const;

    Drawgrid(const std::string& horizontalOffset = "0", const std::string& verticalOffset = "0", const std::string& width = "0", const std::string& height = "0", const std::string& color = "black", const std::string& thickness = "1", bool replace = false);
    virtual ~Drawgrid();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string horizontalOffset_;
    std::string verticalOffset_;
    std::string width_;
    std::string height_;
    std::string color_;
    std::string thickness_;
    bool replace_;
};
