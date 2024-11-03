#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Pixscope : public FilterBase {
public:
    /**
     * Pixel data analysis.
     */
    /**
     * set scope x offset
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setScopeXOffset(float value);
    float getScopeXOffset() const;

    /**
     * set scope y offset
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setScopeYOffset(float value);
    float getScopeYOffset() const;

    /**
     * set scope width
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setScopeWidth(int value);
    int getScopeWidth() const;

    /**
     * set scope height
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setScopeHeight(int value);
    int getScopeHeight() const;

    /**
     * set window opacity
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setWindowOpacity(float value);
    float getWindowOpacity() const;

    /**
     * set window x offset
     * Type: Float
     * Required: No
     * Default: -1.00
     */
    void setWx(float value);
    float getWx() const;

    /**
     * set window y offset
     * Type: Float
     * Required: No
     * Default: -1.00
     */
    void setWy(float value);
    float getWy() const;

    Pixscope(float scopeXOffset = 0.50, float scopeYOffset = 0.50, int scopeWidth = 7, int scopeHeight = 7, float windowOpacity = 0.50, float wx = -1.00, float wy = -1.00);
    virtual ~Pixscope();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float scopeXOffset_;
    float scopeYOffset_;
    int scopeWidth_;
    int scopeHeight_;
    float windowOpacity_;
    float wx_;
    float wy_;
};
