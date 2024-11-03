#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class W3fdif : public FilterBase {
public:
    /**
     * Apply Martin Weston three field deinterlace.
     */
    /**
     * specify the filter
     * Unit: filter
     * Possible Values: simple (0), complex (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setFilter(int value);
    int getFilter() const;

    /**
     * specify the interlacing mode
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

    W3fdif(int filter = 1, int mode = 1, int parity = -1, int deint = 0);
    virtual ~W3fdif();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int filter_;
    int mode_;
    int parity_;
    int deint_;
};
