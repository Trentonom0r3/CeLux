#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Deflicker : public FilterBase {
public:
    /**
     * Remove temporal frame luminance variations.
     */
    /**
     * set how many frames to use
     * Aliases: s
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setSize(int value);
    int getSize() const;

    /**
     * set how to smooth luminance
     * Aliases: m
     * Unit: mode
     * Possible Values: am (0), gm (1), hm (2), qm (3), cm (4), pm (5), median (6)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * leave frames unchanged
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setBypass(bool value);
    bool getBypass() const;

    Deflicker(int size = 5, int mode = 0, bool bypass = false);
    virtual ~Deflicker();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int size_;
    int mode_;
    bool bypass_;
};
