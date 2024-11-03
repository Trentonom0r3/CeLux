#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Morpho : public FilterBase {
public:
    /**
     * Apply Morphological filter.
     */
    /**
     * set morphological transform
     * Unit: mode
     * Possible Values: erode (0), dilate (1), open (2), close (3), gradient (4), tophat (5), blackhat (6)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * when to process structures
     * Unit: str
     * Possible Values: first (0), all (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setStructure(int value);
    int getStructure() const;

    Morpho(int mode = 0, int planes = 7, int structure = 1);
    virtual ~Morpho();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
    int planes_;
    int structure_;
};
