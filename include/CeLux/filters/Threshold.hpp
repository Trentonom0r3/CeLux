#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Threshold : public FilterBase {
public:
    /**
     * Threshold first video stream using other video streams.
     */
    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Threshold(int planes = 15);
    virtual ~Threshold();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
};
