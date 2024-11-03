#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Codecview : public FilterBase {
public:
    /**
     * Visualize information about some codecs.
     */
    /**
     * set motion vectors to visualize
     * Unit: mv
     * Possible Values: pf (1), bf (2), bb (4)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setMv(int value);
    int getMv() const;

    /**
     * 
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setQp(bool value);
    bool getQp() const;

    /**
     * set motion vectors type
     * Aliases: mvt
     * Unit: mv_type
     * Possible Values: fp (1), bp (2)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setMv_type(int value);
    int getMv_type() const;

    /**
     * set frame types to visualize motion vectors of
     * Aliases: ft
     * Unit: frame_type
     * Possible Values: if (1), pf (2), bf (4)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setFrame_type(int value);
    int getFrame_type() const;

    /**
     * set block partitioning structure to visualize
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setBlock(bool value);
    bool getBlock() const;

    Codecview(int mv = 0, bool qp = false, int mv_type = 0, int frame_type = 0, bool block = false);
    virtual ~Codecview();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mv_;
    bool qp_;
    int mv_type_;
    int frame_type_;
    bool block_;
};
