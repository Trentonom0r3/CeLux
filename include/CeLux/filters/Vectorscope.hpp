#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Vectorscope : public FilterBase {
public:
    /**
     * Video vectorscope.
     */
    /**
     * set vectorscope mode
     * Aliases: m
     * Unit: mode
     * Possible Values: gray (0), tint (0), color (1), color2 (2), color3 (3), color4 (4), color5 (5)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set color component on X axis
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setColorComponentOnXAxis(int value);
    int getColorComponentOnXAxis() const;

    /**
     * set color component on Y axis
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setColorComponentOnYAxis(int value);
    int getColorComponentOnYAxis() const;

    /**
     * set intensity
     * Aliases: i
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setIntensity(float value);
    float getIntensity() const;

    /**
     * set envelope
     * Aliases: e
     * Unit: envelope
     * Possible Values: none (0), instant (1), peak (2), peak+instant (3)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setEnvelope(int value);
    int getEnvelope() const;

    /**
     * set graticule
     * Aliases: g
     * Unit: graticule
     * Possible Values: none (0), green (1), color (2), invert (3)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setGraticule(int value);
    int getGraticule() const;

    /**
     * set graticule opacity
     * Aliases: o
     * Type: Float
     * Required: No
     * Default: 0.75
     */
    void setOpacity(float value);
    float getOpacity() const;

    /**
     * set graticule flags
     * Aliases: f
     * Unit: flags
     * Possible Values: white (1), black (2), name (4)
     * Type: Flags
     * Required: No
     * Default: 4
     */
    void setFlags(int value);
    int getFlags() const;

    /**
     * set background opacity
     * Aliases: b
     * Type: Float
     * Required: No
     * Default: 0.30
     */
    void setBgopacity(float value);
    float getBgopacity() const;

    /**
     * set low threshold
     * Aliases: l
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setLthreshold(float value);
    float getLthreshold() const;

    /**
     * set high threshold
     * Aliases: h
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setHthreshold(float value);
    float getHthreshold() const;

    /**
     * set colorspace
     * Aliases: c
     * Unit: colorspace
     * Possible Values: auto (0), 601 (1), 709 (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setColorspace(int value);
    int getColorspace() const;

    /**
     * set 1st tint
     * Aliases: t0
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setTint0(float value);
    float getTint0() const;

    /**
     * set 2nd tint
     * Aliases: t1
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setTint1(float value);
    float getTint1() const;

    Vectorscope(int mode = 0, int colorComponentOnXAxis = 1, int colorComponentOnYAxis = 2, float intensity = 0.00, int envelope = 0, int graticule = 0, float opacity = 0.75, int flags = 4, float bgopacity = 0.30, float lthreshold = 0.00, float hthreshold = 1.00, int colorspace = 0, float tint0 = 0.00, float tint1 = 0.00);
    virtual ~Vectorscope();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
    int colorComponentOnXAxis_;
    int colorComponentOnYAxis_;
    float intensity_;
    int envelope_;
    int graticule_;
    float opacity_;
    int flags_;
    float bgopacity_;
    float lthreshold_;
    float hthreshold_;
    int colorspace_;
    float tint0_;
    float tint1_;
};
