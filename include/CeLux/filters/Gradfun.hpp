#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Gradfun : public FilterBase {
public:
    /**
     * Debands video quickly using gradients.
     */
    /**
     * The maximum amount by which the filter will change any one pixel.
     * Type: Float
     * Required: No
     * Default: 1.20
     */
    void setStrength(float value);
    float getStrength() const;

    /**
     * The neighborhood to fit the gradient to.
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setRadius(int value);
    int getRadius() const;

    Gradfun(float strength = 1.20, int radius = 16);
    virtual ~Gradfun();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float strength_;
    int radius_;
};
