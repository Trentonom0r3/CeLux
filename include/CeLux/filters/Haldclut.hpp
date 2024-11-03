#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Haldclut : public FilterBase {
public:
    /**
     * Adjust colors using a Hald CLUT.
     */
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

    Haldclut(int clut = 1, int interp = 2);
    virtual ~Haldclut();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int clut_;
    int interp_;
};
