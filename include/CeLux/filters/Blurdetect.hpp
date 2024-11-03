#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Blurdetect : public FilterBase {
public:
    /**
     * Blurdetect filter.
     */
    /**
     * set high threshold
     * Type: Float
     * Required: No
     * Default: 0.12
     */
    void setHigh(float value);
    float getHigh() const;

    /**
     * set low threshold
     * Type: Float
     * Required: No
     * Default: 0.06
     */
    void setLow(float value);
    float getLow() const;

    /**
     * search radius for maxima detection
     * Type: Integer
     * Required: No
     * Default: 50
     */
    void setRadius(int value);
    int getRadius() const;

    /**
     * block pooling threshold when calculating blurriness
     * Type: Integer
     * Required: No
     * Default: 80
     */
    void setBlock_pct(int value);
    int getBlock_pct() const;

    /**
     * block size for block-based abbreviation of blurriness
     * Aliases: block_width
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setBlock_height(int value);
    int getBlock_height() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setPlanes(int value);
    int getPlanes() const;

    Blurdetect(float high = 0.12, float low = 0.06, int radius = 50, int block_pct = 80, int block_height = -1, int planes = 1);
    virtual ~Blurdetect();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float high_;
    float low_;
    int radius_;
    int block_pct_;
    int block_height_;
    int planes_;
};
