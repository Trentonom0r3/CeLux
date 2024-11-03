#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Interleave : public FilterBase {
public:
    /**
     * Temporally interleave video inputs.
     */
    /**
     * set number of inputs
     * Aliases: n
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setNb_inputs(int value);
    int getNb_inputs() const;

    /**
     * how to determine the end-of-stream
     * Unit: duration
     * Possible Values: longest (0), shortest (1), first (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDuration(int value);
    int getDuration() const;

    Interleave(int nb_inputs = 2, int duration = 0);
    virtual ~Interleave();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int nb_inputs_;
    int duration_;
};
