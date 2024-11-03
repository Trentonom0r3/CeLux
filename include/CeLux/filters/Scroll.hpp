#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Scroll : public FilterBase {
public:
    /**
     * Scroll input video.
     */
    /**
     * set the horizontal scrolling speed
     * Aliases: h
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setHorizontal(float value);
    float getHorizontal() const;

    /**
     * set the vertical scrolling speed
     * Aliases: v
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setVertical(float value);
    float getVertical() const;

    /**
     * set initial horizontal position
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setHpos(float value);
    float getHpos() const;

    /**
     * set initial vertical position
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setVpos(float value);
    float getVpos() const;

    Scroll(float horizontal = 0.00, float vertical = 0.00, float hpos = 0.00, float vpos = 0.00);
    virtual ~Scroll();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float horizontal_;
    float vertical_;
    float hpos_;
    float vpos_;
};
