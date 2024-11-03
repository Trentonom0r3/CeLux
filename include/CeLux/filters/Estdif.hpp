#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Estdif : public FilterBase {
public:
    /**
     * Apply Edge Slope Tracing deinterlace.
     */
    /**
     * specify the mode
     * Unit: mode
     * Possible Values: frame (0), field (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setMode(int value);
    int getMode() const;

    /**
     * specify the assumed picture field parity
     * Unit: parity
     * Possible Values: tff (0), bff (1), auto (-1)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setParity(int value);
    int getParity() const;

    /**
     * specify which frames to deinterlace
     * Unit: deint
     * Possible Values: all (0), interlaced (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDeint(int value);
    int getDeint() const;

    /**
     * specify the search radius for edge slope tracing
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setRslope(int value);
    int getRslope() const;

    /**
     * specify the search radius for best edge matching
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setRedge(int value);
    int getRedge() const;

    /**
     * specify the edge cost for edge matching
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setEcost(int value);
    int getEcost() const;

    /**
     * specify the middle cost for edge matching
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setMcost(int value);
    int getMcost() const;

    /**
     * specify the distance cost for edge matching
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setDcost(int value);
    int getDcost() const;

    /**
     * specify the type of interpolation
     * Unit: interp
     * Possible Values: 2p (0), 4p (1), 6p (2)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setInterp(int value);
    int getInterp() const;

    Estdif(int mode = 1, int parity = -1, int deint = 0, int rslope = 1, int redge = 2, int ecost = 2, int mcost = 1, int dcost = 1, int interp = 1);
    virtual ~Estdif();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
    int parity_;
    int deint_;
    int rslope_;
    int redge_;
    int ecost_;
    int mcost_;
    int dcost_;
    int interp_;
};
