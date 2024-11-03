#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Mix : public FilterBase {
public:
    /**
     * Mix video inputs.
     */
    /**
     * set number of inputs
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setInputs(int value);
    int getInputs() const;

    /**
     * set weight for each input
     * Type: String
     * Required: No
     * Default: 1 1
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

    /**
     * how to determine end of stream
     * Unit: duration
     * Possible Values: longest (0), shortest (1), first (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDuration(int value);
    int getDuration() const;

    Mix(int inputs = 2, const std::string& weights = "1 1", float scale = 0.00, int planes = 15, int duration = 0);
    virtual ~Mix();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int inputs_;
    std::string weights_;
    float scale_;
    int planes_;
    int duration_;
};
