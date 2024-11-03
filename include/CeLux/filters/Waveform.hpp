#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Waveform : public FilterBase {
public:
    /**
     * Video waveform monitor.
     */
    /**
     * set mode
     * Aliases: m
     * Unit: mode
     * Possible Values: row (0), column (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set intensity
     * Aliases: i
     * Type: Float
     * Required: No
     * Default: 0.04
     */
    void setIntensity(float value);
    float getIntensity() const;

    /**
     * set mirroring
     * Aliases: r
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setMirror(bool value);
    bool getMirror() const;

    /**
     * set display mode
     * Aliases: d
     * Unit: display
     * Possible Values: overlay (0), stack (1), parade (2)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setDisplay(int value);
    int getDisplay() const;

    /**
     * set components to display
     * Aliases: c
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setComponents(int value);
    int getComponents() const;

    /**
     * set envelope to display
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
     * set filter
     * Aliases: f
     * Unit: filter
     * Possible Values: lowpass (0), flat (1), aflat (2), chroma (3), color (4), acolor (5), xflat (6), yflat (7)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFilter(int value);
    int getFilter() const;

    /**
     * set graticule
     * Aliases: g
     * Unit: graticule
     * Possible Values: none (0), green (1), orange (2), invert (3)
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
     * Aliases: fl
     * Unit: flags
     * Possible Values: numbers (1), dots (2)
     * Type: Flags
     * Required: No
     * Default: 1
     */
    void setFlags(int value);
    int getFlags() const;

    /**
     * set scale
     * Aliases: s
     * Unit: scale
     * Possible Values: digital (0), millivolts (1), ire (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setScale(int value);
    int getScale() const;

    /**
     * set background opacity
     * Aliases: b
     * Type: Float
     * Required: No
     * Default: 0.75
     */
    void setBgopacity(float value);
    float getBgopacity() const;

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

    /**
     * set fit mode
     * Aliases: fm
     * Unit: fitmode
     * Possible Values: none (0), size (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFitmode(int value);
    int getFitmode() const;

    /**
     * set input formats selection
     * Unit: input
     * Possible Values: all (0), first (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setInput(int value);
    int getInput() const;

    Waveform(int mode = 1, float intensity = 0.04, bool mirror = true, int display = 1, int components = 1, int envelope = 0, int filter = 0, int graticule = 0, float opacity = 0.75, int flags = 1, int scale = 0, float bgopacity = 0.75, float tint0 = 0.00, float tint1 = 0.00, int fitmode = 0, int input = 1);
    virtual ~Waveform();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
    float intensity_;
    bool mirror_;
    int display_;
    int components_;
    int envelope_;
    int filter_;
    int graticule_;
    float opacity_;
    int flags_;
    int scale_;
    float bgopacity_;
    float tint0_;
    float tint1_;
    int fitmode_;
    int input_;
};
