#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Feedback : public FilterBase {
public:
    /**
     * Apply feedback video filter.
     */
    /**
     * set top left crop position
     * Aliases: x
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setTopLeftCropPosition(int value);
    int getTopLeftCropPosition() const;

    /**
     * set crop size
     * Aliases: w
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCropSize(int value);
    int getCropSize() const;

    Feedback(int topLeftCropPosition = 0, int cropSize = 0);
    virtual ~Feedback();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int topLeftCropPosition_;
    int cropSize_;
};
