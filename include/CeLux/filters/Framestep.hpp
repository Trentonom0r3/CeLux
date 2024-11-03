#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Framestep : public FilterBase {
public:
    /**
     * Select one frame every N frames.
     */
    /**
     * set frame step
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setStep(int value);
    int getStep() const;

    Framestep(int step = 1);
    virtual ~Framestep();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int step_;
};
