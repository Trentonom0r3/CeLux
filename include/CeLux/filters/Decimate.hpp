#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Decimate : public FilterBase {
public:
    /**
     * Decimate frames (post field matching filter).
     */
    /**
     * set the number of frame from which one will be dropped
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setCycle(int value);
    int getCycle() const;

    /**
     * set duplicate threshold
     * Type: Double
     * Required: No
     * Default: 1.10
     */
    void setDupthresh(double value);
    double getDupthresh() const;

    /**
     * set scene change threshold
     * Type: Double
     * Required: No
     * Default: 15.00
     */
    void setScthresh(double value);
    double getScthresh() const;

    /**
     * set the size of the x-axis blocks used during metric calculations
     * Type: Integer
     * Required: No
     * Default: 32
     */
    void setBlockx(int value);
    int getBlockx() const;

    /**
     * set the size of the y-axis blocks used during metric calculations
     * Type: Integer
     * Required: No
     * Default: 32
     */
    void setBlocky(int value);
    int getBlocky() const;

    /**
     * mark main input as a pre-processed input and activate clean source input stream
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setPpsrc(bool value);
    bool getPpsrc() const;

    /**
     * set whether or not chroma is considered in the metric calculations
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setChroma(bool value);
    bool getChroma() const;

    /**
     * set whether or not the input only partially contains content to be decimated
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setMixed(bool value);
    bool getMixed() const;

    Decimate(int cycle = 5, double dupthresh = 1.10, double scthresh = 15.00, int blockx = 32, int blocky = 32, bool ppsrc = false, bool chroma = true, bool mixed = false);
    virtual ~Decimate();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int cycle_;
    double dupthresh_;
    double scthresh_;
    int blockx_;
    int blocky_;
    bool ppsrc_;
    bool chroma_;
    bool mixed_;
};
