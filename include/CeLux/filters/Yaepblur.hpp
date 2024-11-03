#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Yaepblur : public FilterBase {
public:
    /**
     * Yet another edge preserving blur filter.
     */
    /**
     * set window radius
     * Aliases: r
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setRadius(int value);
    int getRadius() const;

    /**
     * set planes to filter
     * Aliases: p
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * set blur strength
     * Aliases: s
     * Type: Integer
     * Required: No
     * Default: 128
     */
    void setSigma(int value);
    int getSigma() const;

    Yaepblur(int radius = 3, int planes = 1, int sigma = 128);
    virtual ~Yaepblur();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int radius_;
    int planes_;
    int sigma_;
};
