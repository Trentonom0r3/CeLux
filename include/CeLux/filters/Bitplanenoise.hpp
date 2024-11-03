#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Bitplanenoise : public FilterBase {
public:
    /**
     * Measure bit plane noise.
     */
    /**
     * set bit plane to use for measuring noise
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setBitplane(int value);
    int getBitplane() const;

    /**
     * show noisy pixels
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setFilter(bool value);
    bool getFilter() const;

    Bitplanenoise(int bitplane = 1, bool filter = false);
    virtual ~Bitplanenoise();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int bitplane_;
    bool filter_;
};
