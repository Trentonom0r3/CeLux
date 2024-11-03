#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Drawbox : public FilterBase {
public:
    /**
     * Draw a colored box on the input video.
     */
    /**
     * set horizontal position of the left box edge
     * Type: String
     * Required: No
     * Default: 0
     */
    void setHorizontalPositionOfTheLeftBoxEdge(const std::string& value);
    std::string getHorizontalPositionOfTheLeftBoxEdge() const;

    /**
     * set vertical position of the top box edge
     * Type: String
     * Required: No
     * Default: 0
     */
    void setVerticalPositionOfTheTopBoxEdge(const std::string& value);
    std::string getVerticalPositionOfTheTopBoxEdge() const;

    /**
     * set width of the box
     * Aliases: w
     * Type: String
     * Required: No
     * Default: 0
     */
    void setWidth(const std::string& value);
    std::string getWidth() const;

    /**
     * set height of the box
     * Aliases: h
     * Type: String
     * Required: No
     * Default: 0
     */
    void setHeight(const std::string& value);
    std::string getHeight() const;

    /**
     * set color of the box
     * Aliases: c
     * Type: String
     * Required: No
     * Default: black
     */
    void setColor(const std::string& value);
    std::string getColor() const;

    /**
     * set the box thickness
     * Aliases: t
     * Type: String
     * Required: No
     * Default: 3
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

    /**
     * use datas from bounding box in side data
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setBox_source(const std::string& value);
    std::string getBox_source() const;

    Drawbox(const std::string& horizontalPositionOfTheLeftBoxEdge = "0", const std::string& verticalPositionOfTheTopBoxEdge = "0", const std::string& width = "0", const std::string& height = "0", const std::string& color = "black", const std::string& thickness = "3", bool replace = false, const std::string& box_source = "");
    virtual ~Drawbox();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string horizontalPositionOfTheLeftBoxEdge_;
    std::string verticalPositionOfTheTopBoxEdge_;
    std::string width_;
    std::string height_;
    std::string color_;
    std::string thickness_;
    bool replace_;
    std::string box_source_;
};
