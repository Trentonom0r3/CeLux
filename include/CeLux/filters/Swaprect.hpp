#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Swaprect : public FilterBase {
public:
    /**
     * Swap 2 rectangular objects in video.
     */
    /**
     * set rect width
     * Type: String
     * Required: No
     * Default: w/2
     */
    void setRectWidth(const std::string& value);
    std::string getRectWidth() const;

    /**
     * set rect height
     * Type: String
     * Required: No
     * Default: h/2
     */
    void setRectHeight(const std::string& value);
    std::string getRectHeight() const;

    /**
     * set 1st rect x top left coordinate
     * Type: String
     * Required: No
     * Default: w/2
     */
    void setX1(const std::string& value);
    std::string getX1() const;

    /**
     * set 1st rect y top left coordinate
     * Type: String
     * Required: No
     * Default: h/2
     */
    void setY1(const std::string& value);
    std::string getY1() const;

    /**
     * set 2nd rect x top left coordinate
     * Type: String
     * Required: No
     * Default: 0
     */
    void setX2(const std::string& value);
    std::string getX2() const;

    /**
     * set 2nd rect y top left coordinate
     * Type: String
     * Required: No
     * Default: 0
     */
    void setY2(const std::string& value);
    std::string getY2() const;

    Swaprect(const std::string& rectWidth = "w/2", const std::string& rectHeight = "h/2", const std::string& x1 = "w/2", const std::string& y1 = "h/2", const std::string& x2 = "0", const std::string& y2 = "0");
    virtual ~Swaprect();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string rectWidth_;
    std::string rectHeight_;
    std::string x1_;
    std::string y1_;
    std::string x2_;
    std::string y2_;
};
