#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Rgbashift : public FilterBase {
public:
    /**
     * Shift RGBA.
     */
    /**
     * shift red horizontally
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setRh(int value);
    int getRh() const;

    /**
     * shift red vertically
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setRv(int value);
    int getRv() const;

    /**
     * shift green horizontally
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setGh(int value);
    int getGh() const;

    /**
     * shift green vertically
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setGv(int value);
    int getGv() const;

    /**
     * shift blue horizontally
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setBh(int value);
    int getBh() const;

    /**
     * shift blue vertically
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setBv(int value);
    int getBv() const;

    /**
     * shift alpha horizontally
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setAh(int value);
    int getAh() const;

    /**
     * shift alpha vertically
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setAv(int value);
    int getAv() const;

    /**
     * set edge operation
     * Unit: edge
     * Possible Values: smear (0), wrap (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setEdge(int value);
    int getEdge() const;

    Rgbashift(int rh = 0, int rv = 0, int gh = 0, int gv = 0, int bh = 0, int bv = 0, int ah = 0, int av = 0, int edge = 0);
    virtual ~Rgbashift();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int rh_;
    int rv_;
    int gh_;
    int gv_;
    int bh_;
    int bv_;
    int ah_;
    int av_;
    int edge_;
};
