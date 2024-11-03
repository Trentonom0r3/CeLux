#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Chromashift : public FilterBase {
public:
    /**
     * Shift chroma.
     */
    /**
     * shift chroma-blue horizontally
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCbh(int value);
    int getCbh() const;

    /**
     * shift chroma-blue vertically
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCbv(int value);
    int getCbv() const;

    /**
     * shift chroma-red horizontally
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCrh(int value);
    int getCrh() const;

    /**
     * shift chroma-red vertically
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCrv(int value);
    int getCrv() const;

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

    Chromashift(int cbh = 0, int cbv = 0, int crh = 0, int crv = 0, int edge = 0);
    virtual ~Chromashift();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int cbh_;
    int cbv_;
    int crh_;
    int crv_;
    int edge_;
};
