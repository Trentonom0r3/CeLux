#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Mestimate : public FilterBase {
public:
    /**
     * Generate motion vectors.
     */
    /**
     * motion estimation method
     * Unit: method
     * Possible Values: esa (1), tss (2), tdls (3), ntss (4), fss (5), ds (6), hexbs (7), epzs (8), umh (9)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setMethod(int value);
    int getMethod() const;

    /**
     * macroblock size
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setMb_size(int value);
    int getMb_size() const;

    /**
     * search parameter
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setSearch_param(int value);
    int getSearch_param() const;

    Mestimate(int method = 1, int mb_size = 16, int search_param = 7);
    virtual ~Mestimate();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int method_;
    int mb_size_;
    int search_param_;
};
