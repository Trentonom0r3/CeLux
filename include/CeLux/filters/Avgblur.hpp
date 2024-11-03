#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Avgblur : public FilterBase {
public:
    /**
     * Apply Average Blur filter.
     */
    /**
     * set horizontal size
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setSizeX(int value);
    int getSizeX() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * set vertical size
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setSizeY(int value);
    int getSizeY() const;

    Avgblur(int sizeX = 1, int planes = 15, int sizeY = 0);
    virtual ~Avgblur();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int sizeX_;
    int planes_;
    int sizeY_;
};
