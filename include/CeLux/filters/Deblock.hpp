#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Deblock : public FilterBase {
public:
    /**
     * Deblock video.
     */
    /**
     * set type of filter
     * Unit: filter
     * Possible Values: weak (0), strong (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setFilter(int value);
    int getFilter() const;

    /**
     * set size of block
     * Type: Integer
     * Required: No
     * Default: 8
     */
    void setBlock(int value);
    int getBlock() const;

    /**
     * set 1st detection threshold
     * Type: Float
     * Required: No
     * Default: 0.10
     */
    void setAlpha(float value);
    float getAlpha() const;

    /**
     * set 2nd detection threshold
     * Type: Float
     * Required: No
     * Default: 0.05
     */
    void setBeta(float value);
    float getBeta() const;

    /**
     * set 3rd detection threshold
     * Type: Float
     * Required: No
     * Default: 0.05
     */
    void setGamma(float value);
    float getGamma() const;

    /**
     * set 4th detection threshold
     * Type: Float
     * Required: No
     * Default: 0.05
     */
    void setDelta(float value);
    float getDelta() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Deblock(int filter = 1, int block = 8, float alpha = 0.10, float beta = 0.05, float gamma = 0.05, float delta = 0.05, int planes = 15);
    virtual ~Deblock();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int filter_;
    int block_;
    float alpha_;
    float beta_;
    float gamma_;
    float delta_;
    int planes_;
};
