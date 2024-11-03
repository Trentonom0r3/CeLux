#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Random : public FilterBase {
public:
    /**
     * Return random frames.
     */
    /**
     * set number of frames in cache
     * Type: Integer
     * Required: No
     * Default: 30
     */
    void setFrames(int value);
    int getFrames() const;

    /**
     * set the seed
     * Type: Integer64
     * Required: No
     * Default: -1
     */
    void setSeed(int64_t value);
    int64_t getSeed() const;

    Random(int frames = 30, int64_t seed = 0);
    virtual ~Random();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int frames_;
    int64_t seed_;
};
