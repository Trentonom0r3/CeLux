#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Lut3d : public FilterBase {
public:
    /**
     * Adjust colors using a 3D LUT.
     */
    /**
     * set 3D LUT file name
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFile(const std::string& value);
    std::string getFile() const;

    /**
     * when to process CLUT
     * Unit: clut
     * Possible Values: first (0), all (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setClut(int value);
    int getClut() const;

    /**
     * select interpolation mode
     * Unit: interp_mode
     * Possible Values: nearest (0), trilinear (1), tetrahedral (2), pyramid (3), prism (4)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setInterp(int value);
    int getInterp() const;

    Lut3d(const std::string& file = "", int clut = 1, int interp = 2);
    virtual ~Lut3d();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string file_;
    int clut_;
    int interp_;
};
