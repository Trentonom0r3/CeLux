#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Deshake : public FilterBase {
public:
    /**
     * Stabilize shaky video.
     */
    /**
     * set x for the rectangular search area
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setXForTheRectangularSearchArea(int value);
    int getXForTheRectangularSearchArea() const;

    /**
     * set y for the rectangular search area
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setYForTheRectangularSearchArea(int value);
    int getYForTheRectangularSearchArea() const;

    /**
     * set width for the rectangular search area
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setWidthForTheRectangularSearchArea(int value);
    int getWidthForTheRectangularSearchArea() const;

    /**
     * set height for the rectangular search area
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setHeightForTheRectangularSearchArea(int value);
    int getHeightForTheRectangularSearchArea() const;

    /**
     * set x for the rectangular search area
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setRx(int value);
    int getRx() const;

    /**
     * set y for the rectangular search area
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setRy(int value);
    int getRy() const;

    /**
     * set edge mode
     * Unit: edge
     * Possible Values: blank (0), original (1), clamp (2), mirror (3)
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setEdge(int value);
    int getEdge() const;

    /**
     * set motion search blocksize
     * Type: Integer
     * Required: No
     * Default: 8
     */
    void setBlocksize(int value);
    int getBlocksize() const;

    /**
     * set contrast threshold for blocks
     * Type: Integer
     * Required: No
     * Default: 125
     */
    void setContrast(int value);
    int getContrast() const;

    /**
     * set search strategy
     * Unit: smode
     * Possible Values: exhaustive (0), less (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setSearch(int value);
    int getSearch() const;

    /**
     * set motion search detailed log file name
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFilename(const std::string& value);
    std::string getFilename() const;

    /**
     * ignored
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setOpencl(bool value);
    bool getOpencl() const;

    Deshake(int xForTheRectangularSearchArea = -1, int yForTheRectangularSearchArea = -1, int widthForTheRectangularSearchArea = -1, int heightForTheRectangularSearchArea = -1, int rx = 16, int ry = 16, int edge = 3, int blocksize = 8, int contrast = 125, int search = 0, const std::string& filename = "", bool opencl = false);
    virtual ~Deshake();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int xForTheRectangularSearchArea_;
    int yForTheRectangularSearchArea_;
    int widthForTheRectangularSearchArea_;
    int heightForTheRectangularSearchArea_;
    int rx_;
    int ry_;
    int edge_;
    int blocksize_;
    int contrast_;
    int search_;
    std::string filename_;
    bool opencl_;
};
