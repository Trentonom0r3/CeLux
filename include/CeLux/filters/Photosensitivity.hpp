#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Photosensitivity : public FilterBase {
public:
    /**
     * Filter out photosensitive epilepsy seizure-inducing flashes.
     */
    /**
     * set how many frames to use
     * Aliases: f
     * Type: Integer
     * Required: No
     * Default: 30
     */
    void setFrames(int value);
    int getFrames() const;

    /**
     * set detection threshold factor (lower is stricter)
     * Aliases: t
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setThreshold(float value);
    float getThreshold() const;

    /**
     * set pixels to skip when sampling frames
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setSkip(int value);
    int getSkip() const;

    /**
     * leave frames unchanged
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setBypass(bool value);
    bool getBypass() const;

    Photosensitivity(int frames = 30, float threshold = 1.00, int skip = 1, bool bypass = false);
    virtual ~Photosensitivity();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int frames_;
    float threshold_;
    int skip_;
    bool bypass_;
};
