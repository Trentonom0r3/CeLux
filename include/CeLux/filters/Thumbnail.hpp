#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Thumbnail : public FilterBase {
public:
    /**
     * Select the most representative frame in a given sequence of consecutive frames.
     */
    /**
     * set the frames batch size
     * Type: Integer
     * Required: No
     * Default: 100
     */
    void setFramesBatchSize(int value);
    int getFramesBatchSize() const;

    /**
     * force stats logging level
     * Unit: level
     * Possible Values: quiet (-8), info (32), verbose (40)
     * Type: Integer
     * Required: No
     * Default: 32
     */
    void setLog(int value);
    int getLog() const;

    Thumbnail(int framesBatchSize = 100, int log = 32);
    virtual ~Thumbnail();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int framesBatchSize_;
    int log_;
};
