#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Tmix : public FilterBase {
public:
    /**
     * Mix successive video frames.
     */
    /**
     * set number of successive frames to mix
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setFrames(int value);
    int getFrames() const;

    /**
     * set weight for each frame
     * Type: String
     * Required: No
     * Default: 1 1 1
     */
    void setWeights(const std::string& value);
    std::string getWeights() const;

    /**
     * set scale
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setScale(float value);
    float getScale() const;

    /**
     * set what planes to filter
     * Type: Flags
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Tmix(int frames = 3, const std::string& weights = "1 1 1", float scale = 0.00, int planes = 15);
    virtual ~Tmix();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int frames_;
    std::string weights_;
    float scale_;
    int planes_;
};
